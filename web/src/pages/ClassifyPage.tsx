/** Flip through candidates, file each one, leave a comment.
 *
 * Laid out like `app_pages/classify.py`, top to bottom:
 *
 *   sidebar   scope (search) · buckets · your comment · others' · filters
 *   header    source id + copy-link icon, nav right
 *   info row  RA/Dec · catalog classifications
 *   viewer    was a PNG of the analysis page; now live charts
 *   links     look up elsewhere
 *   nav bar   Prev · jump · Next · Next unbucketed · progress
 *   keys      left/right mirror Prev/Next
 *
 * `?page=` seeds the selection on first load and is written back as a
 * permalink after; it is never read again, so plain React state stays the
 * single source of truth for which candidate is on screen.
 */
import { Suspense, lazy, useCallback, useEffect, useLayoutEffect, useMemo, useRef,
         useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  ApiError, getCutouts, getGroup, getLightCurve, getSource,
  getSpectra, isAbort, locateSource, nextUnlabelled, putAnnotation,
  putFollowup, reorderLabels, setLabelColor,
} from '../api/client'
import type {
  CatalogSchema, Cutouts, FollowupStatus, GroupMember, LightCurve, SourceDetail,
  SourceQuery,
  Spectra,
  ZtforceResult,
} from '../api/types'
import { LightCurveChart, type EpochClick } from '../charts/LightCurveChart'
import { EpochImageDialog } from '../components/EpochImageDialog'
import { DEFAULT_BIN_WINDOW, ZtforcePanel } from '../components/ZtforcePanel'
import { useSourceWindow } from '../lib/useSourceWindow'
import { useFilters } from '../lib/filterState'
import { FilterBuilder } from '../components/FilterBuilder'
// On demand: it brings Plotly's WebGL stack, which nothing else draws with.
const SpectrumChart = lazy(() =>
  import('../charts/SpectrumChart').then((m) => ({ default: m.SpectrumChart })))
import { WiseColorChart } from '../charts/WiseColorChart'
import { WiseColorCurveChart } from '../charts/WiseColorCurveChart'
import {
  DEFAULT_INFRARED_BANDS, DEFAULT_OPTICAL_SURVEYS, INFRARED_SURVEYS,
} from '../charts/palette'
import { CatalogPanel } from '../components/CatalogPanel'
import { CutoutGrid } from '../components/CutoutGrid'
import { XRayPanel } from '../components/XRayPanel'
import { bandAbsSigma, isTrue, meanAbsSigma } from '../lib/format'
import { useLookahead } from '../lib/useLookahead'
import { PageHeader } from '../components/chrome'
import { Badge, Card, GroupNav, NavBar, SORT_OPTIONS } from '../components/ui'
import { NONE_VALUE, Sidebar } from '../components/Sidebar'
import { imageLinks, tnsLink } from '../lib/links'


/** Light-curve surveys whose points are single exposures we can open. */
const EPOCH_IMAGE_SURVEYS = ['ztf', 'panstarrs']

export function ClassifyPage({ member, schema }:
                             { member: string; schema: CatalogSchema | null }) {
  const [, setParams] = useSearchParams()
  // Which candidate is showing, as an offset into the *filtered* order rather
  // than an index into anything local: the list is ~198k rows and only a
  // window of it is ever in the browser. `selected` is derived from it below.
  // Which order the queue runs in. Defaults to the triage order the sample
  // was built for -- largest change between surveys first -- rather than
  // uuid, which is arbitrary and put the most interesting sources wherever
  // their identifier happened to land.
  const [sort, setSort] = useState<{ key: string; desc: boolean }>(
    { key: SORT_OPTIONS[0].key, desc: SORT_OPTIONS[0].desc })
  const [position, setPosition] = useState(0)
  const [located, setLocated] = useState(false)   // deep link resolved yet?
  // The sidebar's cuts are shared with the Table page, so they live above the
  // routes: narrowing here and switching tabs shows the same sample.
  const {
    catalog, setCatalog,
    columnFilter, setColumnFilter, bucketFilter, deleteBucket, reloadLabels,
    setLabels, setLabelColors, takeLabels, query: sharedQuery, coordListId, coordList,
    ready,
  } = useFilters()
  // A list uploaded with value columns can order the queue by them. They go
  // when the list does, and a sort left pointing at one falls back to the
  // default rather than failing every request.
  const extraSorts = useMemo(() => (coordList?.columns ?? []).map((c) => (
    { key: `list:${c}`, label: c, desc: true })), [coordList])
  useEffect(() => {
    if (sort.key.startsWith('list:') && !extraSorts.some((o) => o.key === sort.key)) {
      setSort({ key: SORT_OPTIONS[0].key, desc: SORT_OPTIONS[0].desc })
    }
  }, [sort.key, extraSorts])

  const [detail, setDetail] = useState<SourceDetail | null>(null)
  const [lc, setLc] = useState<LightCurve | null>(null)
  const [spectra, setSpectra] = useState<Spectra | null>(null)
  const [cutouts, setCutouts] = useState<Cutouts | null>(null)
  // The light-curve point whose exposure is open, if any.
  const [epoch, setEpoch] = useState<(EpochClick & { objectid?: string | null }) | null>(null)
  const [ztf, setZtf] = useState<ZtforceResult | null>(null)
  const [ztfWindow, setZtfWindow] = useState(DEFAULT_BIN_WINDOW)
  // Display stretch for the postage stamps. The renderer takes these as
  // query params, so changing one just points the <img> somewhere else.
  const [stretch, setStretch] = useState('log')
  const [percent, setPercent] = useState(99.5)
  // PanSTARRS stamps binned 4x4 by default: see CutoutGrid's `bin`.
  const [bin, setBin] = useState(4)
  const [comment, setComment] = useState('')
  const [saved, setSaved] = useState('')
  // The comment saves on blur, and blur fires just before a click on a label
  // or nav button runs its own save. Queueing makes the PUTs land in the
  // order they were issued, so a blur carrying the old label can never
  // overwrite the label click that follows it. `sent` is the last note sent
  // for this source, so the blur and the nav flush don't both send it.
  const saveQueue = useRef<Promise<unknown>>(Promise.resolve())
  const sent = useRef('')
  const [flash, setFlash] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  // The vocabulary lives in the filter provider: one copy for both tabs, so
  // switching does not refetch it and a bucket added here shows up there.
  // Optimistic: the list moves as you drop, and the server's answer (or, on
  // a conflict, a fresh copy) replaces it.
  const onReorder = useCallback((next: string[]) => {
    setLabels(next)
    reorderLabels(next)
      .then(takeLabels)
      .catch((e) => {
        setFlash(e instanceof ApiError && e.status === 409
          ? 'Someone changed the buckets meanwhile; reloaded them.'
          : `Reorder failed: ${e}`)
        void reloadLabels()
      })
  }, [reloadLabels, takeLabels])

  // The picker fires on every step of a drag across the palette, so the
  // colour changes on screen at once and is sent once it settles.
  const colorTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})
  const onLabelColor = useCallback((name: string, color: string) => {
    setLabelColors((c) => ({ ...c, [name]: color }))
    clearTimeout(colorTimers.current[name])
    colorTimers.current[name] = setTimeout(() => {
      setLabelColor(name, color).catch((e) => {
        setFlash(`Colour change failed: ${e}`)
        void reloadLabels()
      })
    }, 400)
  }, [reloadLabels])

  // ── the filtered candidate list ────────────────────────────────────────────
  // Not a list any more: a query, a total the server reports, and a window
  // around `position`. `search` is part of the query rather than a local
  // filter -- with only a window in hand, filtering what we hold would
  // silently miss every match outside it.
  const query = useMemo<SourceQuery>(() => ({
    ...sharedQuery,
    sort: sort.key, desc: sort.desc || undefined,
    author: member,
  }), [sharedQuery, sort, member])

  const win = useSourceWindow(query, position)
  const { refresh } = win
  const total = win.total ?? 0
  const current = win.at(position) ?? null
  const selected = current?.source_uuid ?? null

  // A filter or search edit renumbers everything, so go back to the top rather
  // than keeping an offset that now points at an unrelated source. Not until
  // the provider is `ready`, and not for the first query seen after that: the
  // query also changes as the release's schema lands, which used to reset a
  // just-resolved deep link to the top of the list.
  const seenQuery = useRef<SourceQuery | null>(null)
  useEffect(() => {
    if (!ready) return
    if (seenQuery.current === null) { seenQuery.current = query; return }
    if (seenQuery.current === query) return
    seenQuery.current = query
    setPosition(0)
  }, [query, ready])

  // A deep link names a uuid, but navigation works in offsets, so ask the
  // server where that source sits under the current filters. Once only: from
  // then on the URL follows the position, not the other way round.
  // Read once: the effect below rewrites the address bar as soon as any source
  // is on screen, which would otherwise replace the link's target before a retry.
  const deepLink = useRef(new URLSearchParams(window.location.search))
  useEffect(() => {
    // Wait for the query's final shape: an offset found under the
    // provisional one indexes a different order.
    if (located || !ready) return
    const want = deepLink.current.get('page')
    if (!want) { setLocated(true); return }
    const ac = new AbortController()
    locateSource(want, query, ac.signal)
      .then(async (r) => {
        if (r.offset !== null) { setPosition(r.offset); setLocated(true); return }
        // A detection the queue lists under another member of its group: land
        // on that member and show the linked detection in its place.
        if (query.one_per_object) {
          const { members } = await getGroup(want, ac.signal)
          for (const m of members) {
            if (m.source_uuid === want) continue
            const at = await locateSource(m.source_uuid, query, ac.signal)
            if (at.offset !== null) {
              setPosition(at.offset); setLocated(true); setPendingView(want); return
            }
          }
        }
        // Links from before the tab was stored in the URL: widen to every
        // tab once, and the query change re-runs this lookup.
        if (!deepLink.current.has('catalog') && catalog !== 'all') { setCatalog('all'); return }
        setFlash('That source is not in the current selection.')
        setLocated(true)
      })
      .catch((e) => { if (!isAbort(e)) setLocated(true) })
    return () => ac.abort()
  }, [located, ready, query, catalog, setCatalog])

  // Hold the offset inside the range when a filter shrinks the total under it.
  useEffect(() => {
    if (win.total !== null && win.total > 0 && position >= win.total) {
      setPosition(win.total - 1)
    }
  }, [win.total, position])

  // Mirror the selection into the address bar so Copy link keeps working.
  // Write-only, and `replace` keeps Back meaning "leave the page" rather
  // than "undo one candidate".
  //
  // The no-op guard is load-bearing: `setParams` is a fresh function on every
  // location change, so an unguarded write re-triggers this effect through
  // its own dependency and spins forever. Comparing against `window.location`
  // rather than router state keeps the check out of the dependency list --
  // the URL stays an output, it just is not written when it already agrees.
  useEffect(() => {
    if (!selected) return
    const now = new URLSearchParams(window.location.search)
    const filter = columnFilter ? JSON.stringify(columnFilter) : null
    if (now.get('page') === selected && now.get('catalog') === catalog
        && (now.get('filter') ?? null) === filter
        && (now.get('coords') ?? null) === coordListId) return
    setParams({ page: selected, catalog, ...(filter ? { filter } : {}),
                ...(coordListId ? { coords: coordListId } : {}) },
              { replace: true })
  }, [selected, catalog, columnFilter, coordListId, setParams])

  // Navigation is now a clamp on an integer rather than a lookup: the target
  // row may not be in the window yet, and the window follows the position, so
  // moving first and letting the fetch catch up is the whole mechanism.
  const goTo = useCallback((i: number) => {
    // Only the floor is enforced here. Right after a filter change the total
    // is briefly unknown, and clamping against a missing or stale count sends
    // a jump typed in that gap to the wrong end of the list; the effect above
    // pulls the offset back into range as soon as the real count lands.
    const at = Math.max(0, i)
    setPosition(win.total === null ? at : Math.min(at, Math.max(0, win.total - 1)))
  }, [win.total])

  // ── the current source's data ──────────────────────────────────────────────
  // The queue lists each object once; the group toggle can put one of its
  // other detections on screen in place, without moving the queue.
  const [viewing, setViewing] = useState<string | null>(null)
  // A deep link to a non-listed detection, applied once its object is on screen.
  const [pendingView, setPendingView] = useState<string | null>(null)
  useEffect(() => {
    if (pendingView && selected && pendingView !== selected) {
      setViewing(pendingView); setPendingView(null)
    } else setViewing(null)
  }, [selected])  // eslint-disable-line react-hooks/exhaustive-deps
  const uuid = viewing ?? current?.source_uuid
  // Two effects, not one. `stretch` and `percent` are display parameters of
  // the stamps, so folding them in here meant changing either refetched the
  // detail, light curve and spectra, blanked those panels mid-read, dropped
  // the ztforce trace -- and, because this branch reseeds `comment` from the
  // server, discarded a comment typed but not yet blurred.
  useEffect(() => {
    if (!uuid) return
    const ac = new AbortController()
    // Cutouts resolve on their own rather than beside the Promise.all: they
    // are the slowest thing on the page to appear and should not wait for
    // the light curve, and the browser can start fetching the JPEGs the
    // moment the addresses land.
    setCutouts(null)
    getCutouts(uuid, { stretch, percent, bin }, ac.signal).then(setCutouts)
      .catch((e) => { if (!isAbort(e)) console.error(e) })
    return () => ac.abort()
  }, [uuid, stretch, percent, bin])

  useEffect(() => {
    if (!uuid) return
    const ac = new AbortController()
    setDetail(null); setLc(null); setSpectra(null); setZtf(null)
    // Three independent requests, each painting as it lands: bundled in one
    // Promise.all, the classifications and light curve sat blank until the
    // slowest of the three answered.
    const fail = (e: unknown) => { if (!isAbort(e)) setErr(String(e)) }
    getSource(uuid, ac.signal).then((d) => {
      setDetail(d)
      const mine = d.annotations.find((a) => a.author === member)
      setComment(mine?.note ?? '')
      setSaved(mine?.note ?? '')
      sent.current = mine?.note ?? ''
    }).catch(fail)
    getLightCurve(uuid, ac.signal).then(setLc).catch(fail)
    getSpectra(uuid, ac.signal).then(setSpectra).catch(fail)
    return () => ac.abort()
  }, [uuid, member])

  // Other detections of the same object, for the toggle under Prev/Next.
  const [group, setGroup] = useState<GroupMember[]>([])
  useEffect(() => {
    setGroup([])
    if (!selected) return
    const ac = new AbortController()
    getGroup(selected, ac.signal).then((r) => setGroup(r.members))
      .catch((e) => { if (!isAbort(e)) setGroup([]) })
    return () => ac.abort()
  }, [selected])
  // What is on screen: the queue's source, or the group-mate swapped in.
  const shown = viewing
    ? { coord_slug: group.find((m) => m.source_uuid === viewing)?.coord_slug ?? null,
        ra: detail?.ra ?? null, dec: detail?.dec ?? null }
    : { coord_slug: current?.coord_slug ?? null, ra: current?.ra ?? null,
        dec: current?.dec ?? null }

  // Whether the stamps on screen have all settled, credited to the listing
  // the grid is showing -- not the source or stretch being moved to, since
  // for a render after a change the grid still holds the previous (settled)
  // stamps. A new source or stretch is a new listing.
  const [stampsSettled, setStampsSettled] = useState<Cutouts | null>(null)
  const onStampsSettled = useCallback(() => setStampsSettled(cutouts), [cutouts])
  // Then warm the sources around this one while the human reads it (how far:
  // `AHEAD`/`BEHIND` in useLookahead). This, not raw query speed, is what
  // makes flipping feel instant.
  useLookahead(uuid, cutouts !== null && stampsSettled === cutouts,
               (offset) => win.at(position + offset)?.source_uuid, stretch, percent, bin)

  /** The optical curve with the ztforce run folded in as another survey.
   *  `ztforce` is already in OPTICAL_SURVEYS and has a marker in the palette,
   *  so merging here means LightCurveChart draws it with no special case.
   *  The API sends ztforce's stack for every window, so turning the window
   *  redraws without touching the network. */
  // A ZTF point carries its DR objectid, which pins the field/ccd/quadrant the
  // API needs to find the exposure quickly. Looked up by MJD because the
  // drawn points are a catflags-filtered subset of the stored series.
  const onEpochClick = useCallback((e: EpochClick) => {
    const series = lc?.surveys?.[e.survey]?.[e.band]
    const i = series ? series.mjd.findIndex((m) => Math.abs(m - e.mjd) < 1e-7) : -1
    setEpoch({ ...e, objectid: i >= 0 ? series?.catalog_id?.[i] ?? null : null })
  }, [lc])
  // Close only on a real change of source: `uuid` is briefly undefined while
  // the source window refreshes, which is not a reason to drop the panel.
  const epochFor = useRef(uuid)
  useEffect(() => {
    if (uuid && uuid !== epochFor.current) { epochFor.current = uuid; setEpoch(null) }
  }, [uuid])

  const lcWithZtforce = useMemo<LightCurve | null>(() => {
    const binned = ztf?.stacks?.[String(ztfWindow)]
    if (!ztf || !binned || Object.keys(binned).length === 0) return lc
    const base: LightCurve = lc ?? {
      source_uuid: ztf.source_uuid, n_epochs: 0, surveys: {},
    }
    return { ...base, surveys: { ...base.surveys, ztforce: binned } }
  }, [lc, ztf, ztfWindow])

  // Hold the main column at its last settled height while the next source
  // loads. A filter edit swaps the source, the column empties for a moment,
  // and without this the page shrinks, the browser pulls the scroll up, and
  // you land somewhere else when the content comes back.
  const mainRef = useRef<HTMLDivElement | null>(null)
  const heldHeight = useRef(0)
  const loading = !(win.ready && total === 0) && (!current || detail === null || lc === null)
  useLayoutEffect(() => {
    const el = mainRef.current
    if (!el) return
    if (loading) { el.style.minHeight = `${heldHeight.current}px`; return }
    el.style.minHeight = ''
    heldHeight.current = el.offsetHeight
  })

  const mine = detail?.annotations.find((a) => a.author === member)
  const myLabel = mine?.label ?? ''
  // The object's, not mine: one status shared by everyone, and by every
  // detection of the object.
  const status = detail?.followup ?? null
  const statusBy = status ? `set by ${status.author ?? 'someone'} · ${
    new Date(status.updated_at).toLocaleString(
      undefined, { dateStyle: 'medium', timeStyle: 'short' })}` : null

  const shownUuid = useRef(uuid)
  shownUuid.current = uuid
  const commit = useCallback((label: string, note: string) => {
    if (!uuid) return Promise.resolve()
    sent.current = note
    // A PUT replaces the row, so both ride along: saving a label with `note`
    // omitted would clear it.
    const put = saveQueue.current
      .then(() => putAnnotation(uuid, member, { label: label || null, note }))
    saveQueue.current = put.catch(() => {})
    return put.then((a) => {
      // A queued save can finish after you have moved on; only the source
      // still on screen should take its answer.
      if (shownUuid.current === uuid) {
        setDetail((d) => d ? {
          ...d,
          annotations: [...d.annotations.filter((x) => x.author !== member), a],
        } : d)
        setSaved(note)
      }
      // The window is the server's answer, not ours to patch: refetch it so
      // the label, the progress count and (under "unlabelled only") the
      // disappearance of this row all come from the same place.
      refresh()
    }).catch((e) => {
      // Forget the send, so the next blur tries again.
      if (shownUuid.current === uuid && sent.current === note) sent.current = ''
      setFlash(`Save failed: ${e}`)
    })
  }, [uuid, member, refresh])

  // Persist an edited comment before leaving, so an arrow key never discards
  // typed text -- and keep the existing label while doing it.
  const flushComment = useCallback(() => {
    if (comment !== sent.current) void commit(myLabel, comment)
  }, [comment, myLabel, commit])

  // Its own request, not part of the verdict: the status belongs to the
  // object, so it is not queued behind this author's label and note.
  const setStatus = useCallback((st: FollowupStatus | null) => {
    if (!uuid) return
    putFollowup(uuid, member, st).then((fu) => {
      if (shownUuid.current === uuid) setDetail((d) => d ? { ...d, followup: fu } : d)
      refresh()
    }).catch((e) => setFlash(`Status change failed: ${e}`))
  }, [uuid, member, refresh])

  // ── arrow keys mirror Prev/Next ────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const t = e.target as HTMLElement | null
      // Never steal a keystroke from a field.
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA'
                || t.isContentEditable)) return
      if (e.key === 'ArrowLeft' && position > 0) {
        flushComment(); goTo(position - 1)
      } else if (e.key === 'ArrowRight' && position < total - 1) {
        flushComment(); goTo(position + 1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [position, total, goTo, flushComment])

  useEffect(() => {
    if (!flash) return
    const t = setTimeout(() => setFlash(null), 2200)
    return () => clearTimeout(t)
  }, [flash])

  const labelled = win.labelled
  const othersByLabel: Record<string, string[]> = {}
  for (const a of detail?.annotations ?? []) {
    if (a.author !== member && a.label) {
      (othersByLabel[a.label] ??= []).push(a.author)
    }
  }
  const others = (detail?.annotations ?? [])
    .filter((a) => a.author !== member && (a.note ?? '').trim())

  if (err) return <pre style={{ color: 'crimson' }}>{err}</pre>

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '236px 1fr', gap: 18 }}>
      <Sidebar
        mode="classify"
        onLabelColor={onLabelColor}
        onDeleteLabel={(l) => {
          deleteBucket(l).catch((e) => setFlash(String(e instanceof Error ? e.message : e)))
        }}
        currentLabel={myLabel}
        othersByLabel={othersByLabel}
        onReorder={onReorder}
        onLabel={(l) => { void commit(l === myLabel ? '' : l, comment) }}
        currentStatus={status?.status ?? null}
        statusBy={statusBy}
        onStatus={setStatus}
        columnFilters={<FilterBuilder value={columnFilter}
                                      onChange={setColumnFilter}
                                      scope={query} />}
        belowLabels={current && (
          <>
            <label style={{ display: 'block', marginBottom: 4, opacity: 0.8 }}>
              Your comment
            </label>
            <textarea
              value={comment} onChange={(e) => setComment(e.target.value)}
              onBlur={flushComment}
              // Enter leaves the box (and so saves); Shift+Enter is a newline.
              // Not mid-composition, where Enter confirms an IME candidate.
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault(); e.currentTarget.blur()
                }
              }}
              placeholder="What do you make of this source?"
              style={{ width: '100%', height: 110, font: 'inherit',
                       padding: 8, borderRadius: 5, resize: 'vertical',
                       boxSizing: 'border-box',
                       border: '1px solid var(--border-strong)',
                       background: 'var(--bg)', color: 'inherit' }} />
            <div style={{ marginTop: 4, fontSize: 12, opacity: 0.55 }}>
              {comment === saved ? 'Saved' : 'Enter or click away to save · Shift+Enter for a new line'}
            </div>
            {/* Beside your own, so you read what others said while writing. */}
            <div style={{ marginTop: 12 }}>
              <div style={{ opacity: 0.7, marginBottom: 4 }}>
                Comments from others ({others.length})
              </div>
              {others.length === 0 && <div style={{ opacity: 0.5 }}>None yet.</div>}
              {others.map((a) => (
                <Card key={a.author} pad={8} style={{ marginBottom: 6 }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center',
                                flexWrap: 'wrap' }}>
                    <b>{a.author}</b>
                    {a.label && <Badge tone="blue">{a.label}</Badge>}
                  </div>
                  <div style={{ margin: '4px 0', overflowWrap: 'anywhere' }}>
                    {a.note}
                  </div>
                  {/* The raw ISO stamp runs to microseconds and wraps
                      twice at sidebar width. */}
                  <div style={{ opacity: 0.55, fontSize: 12 }}>
                    {a.updated_at ? new Date(a.updated_at).toLocaleString(
                      undefined, { dateStyle: 'medium', timeStyle: 'short' }) : ''}
                  </div>
                </Card>
              ))}
            </div>
          </>
        )}
      />

      <div ref={mainRef} style={{ minWidth: 0 }}>
        {/* The coord slug, not the uuid: it is what the stored plots and
            followup files are named by, so the heading matches what you
            search for on disk. Falls back to the uuid for a source with no
            coordinates, which has no slug.

            Three fixed rows: which object (name, and which of its
            detections), then the queue (Prev/Next, on its own row so it never
            jumps beside or below the name with the window width), then what
            the catalogs say. Navigation stays at the top rather than the foot
            of the page: it is the control you reach for most. */}
        <PageHeader
          title={current
                   ? (shown.coord_slug ?? (uuid ?? current.source_uuid).slice(0, 8))
                   : win.ready && total === 0
                     ? 'No candidates' : ''}
          // Which sub-catalogue this source is from, in the sidebar
          // control's own words, so the two read as the same thing.
          beforeTitle={current?.catalog && (
            // Set in the title's own type (PageHeader's h2: 17px/600) so the
            // tag reads as part of the name rather than a sticker on it.
            <span style={{ fontSize: 17, fontWeight: 600, padding: '1px 7px',
                           borderRadius: 4, whiteSpace: 'nowrap',
                           background: 'rgba(136, 136, 136, 0.18)' }}>
              {current.catalog}
            </span>
          )}
          afterTitle={current && (<>
            <SigmaBadge catalog={detail?.catalog ?? null} />
            <button
              title="Copy link to this source" aria-label="Copy link"
              onClick={() => {
                void copyText(window.location.href).then((ok) => setFlash(ok
                  ? 'Link copied.' : 'Could not copy; the link is in the address bar.'))
              }}
              style={{ display: 'inline-flex', padding: 4, borderRadius: 5,
                       cursor: 'pointer', background: 'none', border: 'none',
                       color: 'inherit', opacity: 0.6 }}>
              {/* Feather's "link" glyph. */}
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2" strokeLinecap="round"
                   strokeLinejoin="round" aria-hidden>
                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
              </svg>
            </button>
            {/* Which detection of this object is on screen: it belongs to
                the name, not to the queue controls below. */}
            <GroupNav members={group} current={uuid ?? current.source_uuid}
              onGo={(target) => {
                flushComment()
                setViewing(target === current.source_uuid ? null : target)
              }} />
          </>)} />

        {current && (
          <div style={{ marginBottom: 10 }}>
            <NavBar
              position={position} total={total} labelled={labelled}
              onPrev={() => { flushComment(); goTo(position - 1) }}
              onNext={() => { flushComment(); goTo(position + 1) }}
              onJump={(i) => { flushComment(); goTo(i) }}
              sort={sort.key} desc={sort.desc} extraSorts={extraSorts}
              onSort={(key, desc) => { flushComment(); setSort({ key, desc }) }}
              onNextUnlabelled={() => {
                flushComment()
                // Under "unlabelled only" every row already qualifies, so
                // the next one is simply the next one -- and asking the
                // server would return an offset in the *other* order, since
                // that endpoint numbers the list without the label filter.
                if (bucketFilter[NONE_VALUE] === 'in') { goTo(position + 1); return }
                // Otherwise ask the server: the next unlabelled source is
                // very often outside the loaded window, and scanning what we
                // hold would report "all done" while 197k rows sit behind it.
                void nextUnlabelled(member, position, query)
                  .then((r) => {
                    if (r.offset === null) {
                      setFlash('Everything in this selection is in a bucket.')
                    } else setPosition(r.offset)
                  })
                  .catch((e) => { if (!isAbort(e)) setFlash(String(e)) })
              }}
            />
          </div>
        )}

        {flash && (
          <div style={{ marginBottom: 8, padding: '6px 10px', borderRadius: 5,
                        background: 'var(--accent-bg)' }}>{flash}</div>
        )}

        {win.ready && total === 0 ? (
          <div style={{ opacity: 0.6 }}>No sources match the current filters.</div>
        ) : !current ? null : (
          <>
            {/* Position, our verdict, and what the cross-matched catalogs
                already call this source. The epoch and spectrum counts used
                to sit here; both are visible in the panels below, where the
                light curve and spectrum either have points or say they do
                not. What is not visible anywhere else is whether SIMBAD or
                DESI has already typed it -- which is usually the first
                thing you want to know. */}
            <div style={{ display: 'flex', gap: 10, alignItems: 'center',
                          flexWrap: 'wrap', marginBottom: 10, opacity: 0.9 }}>
              <span style={{ opacity: 0.75 }}>
                RA, Dec {shown.ra?.toFixed(5)}, {shown.dec?.toFixed(5)}
              </span>
              <Classifications classes={detail?.classes ?? null}
                               redshifts={detail?.redshifts ?? null}
                               classRedshifts={detail?.class_redshifts ?? null}
                               loading={detail === null} />
              <GaiaTag catalog={detail?.catalog ?? null} />
              <XRayTag catalog={detail?.catalog ?? null} />
              {/* With the other classifications: it answers the same question
                  they do -- has anyone already typed this source? */}
              {(() => {
                const t = tnsLink(shown.ra, shown.dec)
                return t && (
                  <a href={t.url} target="_blank" rel="noreferrer" title={t.help}
                     style={{ padding: '2px 8px', borderRadius: 5, fontSize: 12,
                              border: '1px solid var(--border-strong)',
                              textDecoration: 'none', color: 'inherit' }}>
                    {t.label}
                  </a>
                )
              })()}
            </div>

            {/* viewer */}
            <Card>
              <CutoutGrid images={cutouts?.images ?? null}
                          loading={cutouts === null}
                          catalog={detail?.catalog ?? null}
                          pixelArcsec={cutouts?.pixel_arcsec ?? null}
                          regions={cutouts?.regions ?? null}
                          stretch={stretch} onStretch={setStretch}
                          percent={percent} onPercent={setPercent}
                          bin={bin} onBin={setBin}
                          links={imageLinks(shown.ra, shown.dec)}
                          onSettled={onStampsSettled} />
              <LightCurveChart lightcurve={lcWithZtforce} title="Optical"
                               defaultSurveys={DEFAULT_OPTICAL_SURVEYS}
                               yAxisTitle="Mag"
                               clickableSurveys={EPOCH_IMAGE_SURVEYS}
                               onEpochClick={onEpochClick} />
              {epoch && uuid && (
                <EpochImageDialog sourceUuid={uuid} epoch={epoch}
                                  referenceUrl={cutouts?.images?.[epoch.survey]?.[epoch.band] ?? null}
                                  referencePixelArcsec={cutouts?.pixel_arcsec?.[epoch.survey] ?? null}
                                  stretch={stretch} percent={percent}
                                  onClose={() => setEpoch(null)} />
              )}
              <ZtforcePanel key={uuid ?? 'none'} sourceUuid={uuid ?? null}
                            onResult={setZtf}
                            window={ztfWindow} onWindow={setZtfWindow} />
              {/* 2:1, matching the analysis page's 8-and-4 of its 26-column
                  grid. The colour histogram answers the question the WISE
                  curve raises, so they belong side by side. */}
              {/* Two rows so the X-ray box can centre on the colour curve beside it. */}
              <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr',
                            columnGap: 14, alignItems: 'start' }}>
                <LightCurveChart lightcurve={lc} surveys={INFRARED_SURVEYS}
                                 defaultBands={DEFAULT_INFRARED_BANDS}
                                 title="WISE / NEOWISE" yAxisTitle="Vega Mag" />
                <WiseColorChart lightcurve={lc} />
                <WiseColorCurveChart lightcurve={lc} />
                <div style={{ alignSelf: 'center' }}>
                  <XRayPanel catalog={detail?.catalog ?? null}
                             ra={shown.ra} dec={shown.dec} />
                </div>
              </div>
              <Suspense fallback={<div style={{ height: 60 }} />}>
                <SpectrumChart spectra={spectra} />
              </Suspense>
            </Card>

            {/* Open, not behind a disclosure: it is reference you read
                while classifying, not an aside. */}
            <div style={{ marginTop: 14 }}>
              <CatalogPanel schema={schema} catalog={detail?.catalog ?? null} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}


/** An X-ray tag when eROSITA DR2 has a source within 10"; nothing otherwise,
 *  as with the other catalog matches. */
function XRayTag({ catalog }: { catalog: Record<string, unknown> | null }) {
  if (!catalog || !isTrue(catalog.erosita_matched)) return null
  const sep = catalog.erosita_sep_arcsec
  return (
    <span title={typeof sep === 'number' ? `eROSITA DR2 source ${sep.toFixed(1)}" away` : undefined}
          style={{ display: 'inline-flex', gap: 5, alignItems: 'baseline',
                   padding: '2px 8px', borderRadius: 999, fontSize: 12,
                   background: 'var(--class-bg)',
                   border: '1px solid var(--class-border)' }}>
      <b style={{ color: 'var(--class)' }}>X-ray</b>
    </span>
  )
}

/** Whether Gaia has this source, and whether it published a pm/parallax for it.
 *
 * Read from the `gaia_*` columns `filter_fields.annotate_gaia` writes. A match
 * with no astrometry (a 2p solution) passes the pm/parallax cuts by default, so
 * it is flagged "(no fit)". Absent for releases built before those columns.
 */
function GaiaTag({ catalog }: { catalog: Record<string, unknown> | null }) {
  if (!catalog || !isTrue(catalog.gaia_matched)) return null
  const fit = isTrue(catalog.gaia_has_astrometry)
  const num = (k: string) => {
    const v = catalog[k]
    return typeof v === 'number' && Number.isFinite(v) ? v : null
  }
  const sep = num('gaia_sep_arcsec')
  const poe = num('gaia_parallax_over_error')
  const pmSig = num('gaia_pm_sig')
  const help = [
    sep !== null ? `${sep.toFixed(2)}" from Gaia source` : null,
    fit ? `parallax/σ ${poe?.toFixed(1) ?? '–'}, pm/σ ${pmSig?.toFixed(1) ?? '–'}`
        : 'position-only solution: no pm or parallax published',
  ].filter(Boolean).join(' · ')
  return (
    <span title={help}
          style={{ display: 'inline-flex', gap: 5, alignItems: 'baseline',
                   padding: '2px 8px', borderRadius: 999, fontSize: 12,
                   background: 'var(--class-bg)',
                   border: '1px solid var(--class-border)' }}>
      <b style={{ color: 'var(--class)' }}>Gaia</b>
      {!fit && <span style={{ opacity: 0.75 }}>(no fit)</span>}
    </span>
  )
}

/** Mean |Δmag σ| beside the name: the value the queue is sorted by. "≥" when a
 *  band hit the envelope's cap, so the mean is a lower bound. */
function SigmaBadge({ catalog }: { catalog: Record<string, unknown> | null }) {
  const m = meanAbsSigma(catalog)
  if (!m) return null
  const help = ['g', 'r', 'i']
    .map((b) => [b, bandAbsSigma(catalog ?? {}, b)] as const)
    .filter(([, v]) => v !== null)
    .map(([b, v]) => `${b}: ${(v as number).toFixed(2)}σ`).join(' · ')
  return (
    <span title={`Mean |Δmag σ| (${help})`}
          style={{ fontSize: 17, fontWeight: 600, opacity: 0.6, whiteSpace: 'nowrap',
                   fontVariantNumeric: 'tabular-nums' }}>
      |σ| {m.capped ? '≥' : ''}{m.value.toFixed(2)}
    </span>
  )
}

/** Copy to the clipboard. `navigator.clipboard` only exists on https or
 *  localhost, so plain-http access falls back to the old execCommand route. */
async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard) { await navigator.clipboard.writeText(text); return true }
  } catch { /* permission denied or not focused: try the fallback */ }
  const ta = document.createElement('textarea')
  ta.value = text
  ta.style.position = 'fixed'
  ta.style.opacity = '0'
  document.body.appendChild(ta)
  ta.select()
  try { return document.execCommand('copy') }
  catch { return false }
  finally { document.body.removeChild(ta) }
}



/** What the cross-matched catalogs call this source.
 *
 * Derived server-side (`api/derive.classes_of`) so this and the table cannot
 * disagree. A catalog that matched but never typed the source reads
 * `unknown`, which is a different statement from being absent here
 * entirely, and is why the empty case says "no known classifications"
 * rather than showing nothing at all.
 */
function Classifications({ classes, redshifts, classRedshifts, loading }:
                         { classes: Record<string, string[]> | null
                           redshifts: Record<string, number> | null
                           classRedshifts: Record<string, Record<string, number>> | null
                           loading: boolean }) {
  if (loading) return <span style={{ opacity: 0.4 }}>·</span>
  const entries = Object.entries(classes ?? {})
  if (entries.length === 0) {
    return (
      <span style={{ opacity: 0.55, fontStyle: 'italic' }}>
        no known classifications
      </span>
    )
  }
  return (
    <span style={{ display: 'flex', gap: 6, flexWrap: 'wrap',
                   alignItems: 'center' }}>
      {entries.map(([catalog, values]) => {
        // A catalog can type the source without giving a redshift, so this is
        // absent far more often than the class is -- rendered only when
        // present rather than as an empty "(z=)".
        const z = redshifts?.[catalog]
        // SDSS: every spectrum of the source counts, so each class carries
        // its own redshift.
        const perClass = classRedshifts?.[catalog]
        return (
          <span key={catalog}
                style={{ display: 'inline-flex', gap: 5, alignItems: 'baseline',
                         padding: '2px 8px', borderRadius: 999, fontSize: 12,
                         background: 'var(--class-bg)',
                         border: '1px solid var(--class-border)' }}>
            <span style={{ opacity: 0.6 }}>{catalog}</span>
            <b style={{ color: 'var(--class)' }}>
              {perClass
                ? values.map((v, i) => (
                    <span key={v}>
                      {i > 0 && ' / '}{v}
                      {typeof perClass[v] === 'number' && (
                        <span style={{ fontWeight: 400, opacity: 0.75 }}>
                          {' '}(z={perClass[v].toFixed(4)})
                        </span>
                      )}
                    </span>
                  ))
                : values.join(' / ')}
            </b>
            {!perClass && typeof z === 'number' && Number.isFinite(z) && (
              // 4 decimals to match the `:.4f` the analysis-page info panel
              // already uses for the AGN-DB and SIMBAD redshifts.
              <span style={{ opacity: 0.75 }}>(z={z.toFixed(4)})</span>
            )}
          </span>
        )
      })}
    </span>
  )
}
