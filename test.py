from MDAnalysis import Universe

md_traj = Universe('./example/md0.pdb', './example/md0.xtc')
# print(md_traj)
# print(md_traj.trajectory[0].positions.shape)
print(md_traj.trajectory[0].positions[100])
print(md_traj.trajectory[1].positions[100])
print(md_traj.trajectory[5].positions[100])
print(md_traj.trajectory[5].positions[200])
# help(md_traj.trajectory[0])