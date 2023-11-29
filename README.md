# pydefect_2d

## Workflow for 2D Point Defect Calculations

1. Generate `unitcell.yaml` using pydefect

1. Create dielectric profile using either gdd or sdd subcommand.
``` pydefect_2d sdd -c 0.5 -s 0.5 -w 7.15 -wz 7.15 -u unitcell.yaml -pl ../../defects/6_30A/perfect/LOCPOT --denominator 2```

1. Perform standard defect calculations with pydefect.

1. Create the defects/correction directory.

1. Generate the 1d_gauss directory and create **gauss1_d_potential_xxxx.json** using the following command:
 ```pydefect_2d 1gm -s ../../supercell_info.json -r 0.3 0.5 -dd ../dielectric_const_dist.json```

1. Calculate Gaussian charge energy under 3D periodic boundary conditions and in isolated conditions.
```pydefect_2d gmz -z 0.3{0,2,4,6,8} 0.4{0,2,4,6,8} 0.5 -s ../../supercell_info.json -cd . -dd ../dielectric_const_dist.json```

1.  Generate **gauss_energies.json** inside defects/correction/.
```pydefect_2d ge```

1. Compute the one-dimensional potential from first-principles calculations and determine the Gaussian charge center.
```pydefect_2d 1fp -d . -pl ../../perfect/LOCPOT -od ../1d_gauss```

1. Generate slab_model.json and correction.json (at this point, it converges with pydefect).
```pydefect_2d 1sm -d Va_MoS6_-2 -dd dielectric_const_dist.json -od 1d_gauss -g correction/gauss_energies.json```
