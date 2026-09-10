from pymol_remote.client import PymolSession
cmd = PymolSession(hostname="127.0.0.1", port=9123)
pdb_txt = open("/mnt/netapp1/Store_othcxlwa/FRUTON-NEW/11UE/11UE.pdb").read()
print("file length:", len(pdb_txt), "chars")
print("first line:", pdb_txt.splitlines()[0] if pdb_txt else "(empty)")
result = cmd.read_pdbstr(pdb_txt, "p11ue")
print("read_pdbstr returned:", result)
print("names now:", cmd.get_names())
print("atoms in p11ue:", cmd.count_atoms("p11ue"))
  