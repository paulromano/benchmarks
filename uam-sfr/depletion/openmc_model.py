from math import pi, sqrt
import json

import numpy as np
import openmc
import openmc.deplete

################################################################################
# DIMENSIONS FROM LARGE OXIDE CASE

temperature_fuel = 1227 + 273.15
temperature_structure = 470 + 273.15
subassembly_pitch = 21.2205
subassembly_duct_outer = 20.7468
subassembly_duct_thickness = 0.4525
subassembly_duct_inner = subassembly_duct_outer - 2*subassembly_duct_thickness

inner_hole_radius = 0.1257
fuel_radius = 0.4742
clad_inner_radius = 0.4893
clad_outer_radius = 0.5419

pin_pitch = 1.1897

################################################################################
# MATERIALS

# Fuel material from table 24
upuo2 = openmc.Material(name='(U,Pu)O2', temperature=temperature_fuel)
upuo2.add_element('O', 4.2825e-02)
upuo2.add_nuclide('U234', 1.7602e-06)
upuo2.add_nuclide('U235', 3.3412e-05)
upuo2.add_nuclide('U236', 4.0736e-06)
upuo2.add_nuclide('U238', 1.8692e-02)
upuo2.add_nuclide('Np237', 3.7863e-06)
upuo2.add_nuclide('Np239', 3.5878e-06)
upuo2.add_nuclide('Pu238', 9.4366e-05)
upuo2.add_nuclide('Pu239', 1.8178e-03)
upuo2.add_nuclide('Pu240', 1.0177e-03)
upuo2.add_nuclide('Pu241', 2.1797e-04)
upuo2.add_nuclide('Pu242', 3.2651e-04)
upuo2.add_nuclide('Am241', 4.0395e-05)
upuo2.add_nuclide('Am242', 1.2900e-08)
upuo2.add_nuclide('Am242_m1', 1.2243e-06)
upuo2.add_nuclide('Am243', 2.4048e-05)
upuo2.add_nuclide('Cm242', 2.2643e-06)
upuo2.add_nuclide('Cm243', 1.0596e-07)
upuo2.add_nuclide('Cm244', 3.2454e-06)
upuo2.add_nuclide('Cm245', 1.4745e-07)
upuo2.add_nuclide('Cm246', 3.4906e-09)
upuo2.add_element('Mo', 2.7413e-03)
upuo2.volume = 271*pi*(fuel_radius**2 - inner_hole_radius**2)

# Structure and coolant materials from table 9
em10 = openmc.Material(name='EM10')
em10.add_element('C', 3.8254e-04)
em10.add_element('Si', 4.9089e-04)
em10.add_element('Ti', 1.9203e-05)
em10.add_element('Cr', 7.5122e-03)
em10.add_element('Fe', 7.3230e-02)
em10.add_element('Ni', 3.9162e-04)
em10.add_element('Mo', 4.7925e-04)
em10.add_element('Mn', 4.1817e-04)

ods = openmc.Material(name='ODS')
ods.add_element('C', 3.5740e-04)
ods.add_element('O', 3.9924e-04)
ods.add_element('Ti', 5.3824e-04)
ods.add_element('Cr', 1.7753e-02)
ods.add_element('Fe', 5.3872e-02)
ods.add_element('Ni', 3.6588e-04)
ods.add_element('Mn', 2.3441e-04)
ods.add_element('P', 2.7718e-05)
ods.add_element('Al', 9.1482e-03)
ods.add_element('Co', 2.1852e-04)
ods.add_element('Cu', 1.0135e-04)
ods.add_element('Y', 2.6616e-04)

sodium = openmc.Material(name='Na')
sodium.add_element('Na', 2.1924e-02)

helium = openmc.Material()
helium.add_element('He', 1.0e-6)

materials = openmc.Materials([upuo2, helium, ods, sodium, em10])
materials.export_to_xml()

################################################################################
# GEOMETRY

fuel_inner = openmc.ZCylinder(r=inner_hole_radius)
fuel_outer = openmc.ZCylinder(r=fuel_radius)
clad_inner = openmc.ZCylinder(r=clad_inner_radius)
clad_outer = openmc.ZCylinder(r=clad_outer_radius)

inner_hole = openmc.Cell(fill=helium, region=-fuel_inner)
fuel = openmc.Cell(fill=upuo2, region=+fuel_inner & -fuel_outer)
gap = openmc.Cell(fill=helium, region=+fuel_outer & -clad_inner)
clad = openmc.Cell(fill=ods, region=+clad_inner & -clad_outer)
outside_pin = openmc.Cell(fill=sodium, region=+clad_outer)

pin_universe = openmc.Universe(cells=(inner_hole, fuel, gap, clad, outside_pin))

na_cell = openmc.Cell(fill=sodium)
na_universe = openmc.Universe(cells=(na_cell,))

lattice = openmc.HexLattice()
lattice.center = (0., 0.)
lattice.pitch = (pin_pitch,)
lattice.orientation = 'x'
lattice.universes = [[pin_universe]]
lattice.universes = [
    [pin_universe for _ in range(max(1, 6*ring_index))]
    for ring_index in reversed(range(10))
]
lattice.outer = na_universe

outer_hex = openmc.model.hexagonal_prism(
    subassembly_pitch / sqrt(3.),
    orientation='x',
    boundary_type='periodic'
)
duct_outer_hex = openmc.model.hexagonal_prism(
    subassembly_duct_outer / sqrt(3.), orientation='x')
duct_inner_hex = openmc.model.hexagonal_prism(
    subassembly_duct_inner / sqrt(3.), orientation='x')

lattice_cell = openmc.Cell(fill=lattice, region=duct_inner_hex)
duct = openmc.Cell(fill=em10, region=~duct_inner_hex & duct_outer_hex)
outside_duct = openmc.Cell(fill=sodium, region=~duct_outer_hex & outer_hex)

geom = openmc.Geometry([lattice_cell, duct, outside_duct])
geom.export_to_xml()

################################################################################
# SIMULATION SETTINGS

settings = openmc.Settings()
settings.particles = 10000
settings.inactive = 20
settings.batches = 100
settings.temperature = {
    'default': temperature_structure,
    'method': 'interpolation'
}
settings.export_to_xml()

################################################################################
# DEPLETION

cumulative_days = np.array([0.0, 102.5, 205.0, 307.5, 410.0])
timesteps = np.diff(cumulative_days)
power_density = 50.4  # MW/tHM = W/gHM

# Get Serpent fission Q values
with open('serpent_fissq.json') as fh:
    serpent_q = json.load(fh)

# Create depletion operator
op = openmc.deplete.Operator(geom, settings, chain_file='chain_casl_sfr.xml',
                             fission_q=serpent_q)

# Execute depletion using integrator
integrator = openmc.deplete.PredictorIntegrator(
    op, timesteps, power_density=power_density, timestep_units='d')
integrator.integrate()
