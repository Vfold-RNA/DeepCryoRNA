import sys
from chimerax.core.commands import run

print(f"command for preprocessing cryo-EM map: {' '.join(sys.argv)}")

cryoEM_map, contour = sys.argv[3:5]
contour = float(contour)
if contour < 0.:
    new_contour = contour
else:
    new_contour = contour*0.5

if cryoEM_map.endswith(".map"):
    map_name = cryoEM_map.split("/")[-1].split(".map")[0]
elif cryoEM_map.endswith(".mrc"):
    map_name = cryoEM_map.split("/")[-1].split(".mrc")[0]
else:
    raise ValueError(f"CryoEM map name should end with '.map' or '.mrc'.")

run(session,f"volume showPlane false")
run(session,f"open {cryoEM_map}")
run(session,f"volume #1 style surface level {new_contour} step 1")
run(session,f"volume #1 calculateSurface true")
run(session,f"surface dust #1 size 25")
run(session,f"volume mask #1 surfaces #1")
run(session,f"save tmp/{map_name}_masked.mrc #2")
run(session,"close all")

run(session,f"open tmp/{map_name}_masked.mrc")
run(session,"vol resample #1 spacing 0.5")
run(session,f"volume #2 style surface level {new_contour} step 1")
run(session,f"volume #2 calculateSurface true")
run(session,f"surface dust #2 size 25")
run(session,f"volume mask #2 surfaces #2")
run(session,f"volume #3 style surface level {new_contour} step 1")
run(session,f"volume #3 calculateSurface true")
run(session,f"surface dust #3 size 25")
run(session,f"save tmp/{map_name}_masked_apix0.5.mrc #3")
run(session,"close all")

run(session,"quit")
