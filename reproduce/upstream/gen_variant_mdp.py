#!/usr/bin/env python3
"""Apply a config's mdp overrides on top of a per-complex baseline production mdp.
Usage: gen_variant_mdp.py <config_id> <base.mdp> <out.mdp> [nsteps_override]
"""
import sys, re
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import md_configs

def parse_mdp(path):
    order, vals = [], {}
    for line in open(path):
        raw = line.rstrip("\n")
        m = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=\s*(.*?)\s*$", raw)
        if not m:
            continue
        key = m.group(1).strip().lower()
        if key not in vals:
            order.append(key)
        vals[key] = m.group(2).strip()
    return order, vals

def main():
    cfg_id, base, out = sys.argv[1], sys.argv[2], sys.argv[3]
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else None
    cfg = md_configs.get(cfg_id, nsteps=nsteps)
    order, vals = parse_mdp(base)
    for k, v in cfg["overrides"].items():
        k = k.lower()
        if k not in vals:
            order.append(k)
        vals[k] = str(v)
    with open(out, "w") as f:
        f.write(f"; variant config: {cfg_id}  ({cfg['note']})\n")
        for k in order:
            f.write(f"{k:28s} = {vals[k]}\n")
    print(f"wrote {out}  [{cfg_id}] dt={vals.get('dt')} nsteps={vals.get('nsteps')} "
          f"mts={vals.get('mts')} rc={vals.get('rcoulomb')} rvdw={vals.get('rvdw')} "
          f"nstlist={vals.get('nstlist')} integ={vals.get('integrator')}")

if __name__ == "__main__":
    main()
