import time
from smi_browser import nsls2api
import tiled_browser as tb
from tiled.queries import Key

def run_benchmarks():
    # 1. Scale check
    print("--- Scale Check ---")
    for cycle in ['2026-2', '2025-2']:
        t0 = time.perf_counter()
        proposals = nsls2api.fetch_proposals_for_cycle(cycle)
        dt = time.perf_counter() - t0
        print(f'fetch_proposals_for_cycle({cycle}): {len(proposals)} proposals in {dt:.3f}s')
        if proposals:
            print(f'First 10: {proposals[:10]}')
        print()

    # 2. Full pipeline
    cat = tb.connect()
    cycle = '2026-2'
    print(f'=== Full pipeline: cycle {cycle} ===')
    
    t0 = time.perf_counter()
    proposal_ids = nsls2api.fetch_proposals_for_cycle(cycle)
    t_api = time.perf_counter() - t0
    print(f'Step 1 - API fetch cycle proposals: {t_api:.3f}s -> {len(proposal_ids)} proposals')

    t0 = time.perf_counter()
    found = []
    for pid in proposal_ids:
        ds = f'pass-{pid}'
        sub = cat.search(Key('data_session') == ds)
        n = len(sub)
        if n > 0:
            found.append((ds, n))

    t_tiled = time.perf_counter() - t0
    print(f'Step 2 - Check {len(proposal_ids)} proposals in tiled: {t_tiled:.3f}s')
    print(f'  Found {len(found)} proposals with data in tiled')
    print(f'  Total time: {t_api + t_tiled:.3f}s')
    print()
    for ds, n in found[:15]:
        print(f'  {ds}: {n} scans')

if __name__ == "__main__":
    run_benchmarks()
