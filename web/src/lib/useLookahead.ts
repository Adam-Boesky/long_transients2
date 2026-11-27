/** Warms the browser's HTTP cache for the sources around the one on screen.
 *
 * This is what makes paging feel instant: by the time you press Next (or
 * Prev) the stamps and light curve are cached and resolve from memory. The
 * cutout stamps are the slow part, rendered on demand from IRSA and STScI.
 *
 * - Starts once the source on screen has its stamps (`ready`), or after
 *   `FALLBACK_MS` if one hangs, so it never competes with them.
 * - Nearest first, one source at a time, so the next source finishes before
 *   anything further out starts.
 * - Every URL is fetched at most once: one that settled is never asked for
 *   again, and one in flight is shared rather than re-requested.
 * - Moving on aborts what is no longer near, but never the source now on
 *   screen -- its `<img>` may be waiting on that same response.
 * - Only immutable responses (stamps, light curve) are warmed: anything else
 *   would be fetched again on arrival anyway.
 *
 * Upstream, `prefetchUrl` marks these as background work, so the API keeps
 * IRSA headroom for the page being read.
 */
import { useEffect, useRef } from 'react'
import { getCutouts, isAbort, lightCurveUrl, prefetchUrl } from '../api/client'

type Flight = { ac: AbortController; done: Promise<void>; owner: string }

// Generous: a URL is a few dozen bytes, and forgetting one only costs a
// cache hit, never a duplicate upstream fetch.
const MAX_SETTLED = 5000
const FALLBACK_MS = 8000
// How far to warm: sources ahead (Next) and behind (Prev) of the one shown.
const AHEAD = 5
const BEHIND = 1

/** Offsets to warm, nearest first, alternating ahead and behind: 1, -1, 2, 3... */
const OFFSETS = Array.from({ length: Math.max(AHEAD, BEHIND) }, (_, i) => i + 1)
  .flatMap((d) => [...(d <= AHEAD ? [d] : []), ...(d <= BEHIND ? [-d] : [])])

/** `neighbour(offset)` is the source that many places from the one shown. */
export function useLookahead(current: string | undefined, ready: boolean,
                             neighbour: (offset: number) => string | undefined,
                             stretch: string, percent: number, bin: number = 1) {
  const settled = useRef(new Set<string>())
  const flights = useRef(new Map<string, Flight>())
  // A primitive, so a new array with the same sources does not restart it.
  const order = OFFSETS.map(neighbour).filter((u): u is string => !!u)
  const orderKey = order.join(',')

  useEffect(() => {
    // Stamps differ per stretch, so the owner of a URL carries it.
    const owner = (uuid: string) => `${uuid}|${stretch}|${percent}|${bin}`
    const keep = new Set([...order, ...(current ? [current] : [])].map(owner))
    for (const [url, f] of flights.current) {
      if (!keep.has(f.owner)) {
        f.ac.abort()
        flights.current.delete(url)
      }
    }

    const warm = (url: string, uuid: string): Promise<void> => {
      if (settled.current.has(url)) return Promise.resolve()
      const inFlight = flights.current.get(url)
      if (inFlight) return inFlight.done
      const ac = new AbortController()
      const done = prefetchUrl(url, ac.signal)
        // A failure settles it too: the page's own request is the retry.
        .then(() => true, (e: unknown) => !isAbort(e))
        .then((finished) => {
          if (finished) {
            settled.current.add(url)
            if (settled.current.size > MAX_SETTLED) {
              settled.current.delete(settled.current.values().next().value as string)
            }
          }
          if (flights.current.get(url)?.ac === ac) flights.current.delete(url)
        })
      flights.current.set(url, { ac, done, owner: owner(uuid) })
      return done
    }

    let stale = false
    const run = async () => {
      for (const uuid of order) {
        let urls = [lightCurveUrl(uuid)]
        try {
          const c = await getCutouts(uuid, { stretch, percent, bin })
          urls = urls.concat(Object.values(c.images).flatMap((bands) => Object.values(bands)))
        } catch { /* the page reports it if this source is opened */ }
        if (stale) return
        await Promise.all(urls.map((u) => warm(u, uuid)))
        if (stale) return
      }
    }
    const timer = globalThis.setTimeout(() => void run(), ready ? 0 : FALLBACK_MS)
    return () => { stale = true; globalThis.clearTimeout(timer) }
  }, [current, ready, orderKey, stretch, percent, bin])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => {
    for (const f of flights.current.values()) f.ac.abort()
    flights.current.clear()
  }, [])
}
