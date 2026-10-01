"""Reproducible per-lane mix of fresh sessions and genuine follow-up turns."""
import asyncio
from dataclasses import asdict
import json
import random
import time

START_POSITIONS=(0,1,2,9,16,23,30,37,44,51)

def make_schedule(seed=20261001):
    lanes=[]
    for lane in range(8):
        rng=random.Random(seed+lane);turns=[0]*10;started=0;rows=[]
        for position in range(60):
            if position in START_POSITIONS:
                slot=started;started+=1
            else:
                candidates=[s for s in range(started) if turns[s]<6]
                slot=rng.choice(candidates)
            rows.append(dict(lane=lane,position=position,session_index=lane*10+slot,
                             turn=turns[slot],new_session=turns[slot]==0))
            turns[slot]+=1
        assert turns==[6]*10
        lanes.append(rows)
    return dict(seed=seed,start_positions=list(START_POSITIONS),lanes=lanes,
                requests=480,concurrency=8,sessions=80,turns_per_session=6)

async def run_mixed(up,args,schedule,dispatch_path):
    import httpx
    import openai
    sessions=[up.ChatSession(args) for _ in range(80)]
    async with httpx.AsyncClient(timeout=args.timeout,verify=not args.skip_ssl_verify) as http:
        async def lane_run(rows):
            client=openai.AsyncOpenAI(base_url=f'{args.base_url}/v1',api_key='EMPTY',http_client=http)
            results=[]
            for row in rows:
                session=sessions[row['session_index']]
                assert session.turns==row['turn']
                session.mixed_schedule_row=row
                with dispatch_path.open('a') as f:
                    f.write(json.dumps(dict(**row,time_ns=time.time_ns()))+'\n')
                result=await up.run_turn(session,client,http,args.base_url,0)
                results.append(asdict(result))
            return results
        results=await asyncio.gather(*(lane_run(rows) for rows in schedule['lanes']))
    return [r for lane in results for r in lane]
