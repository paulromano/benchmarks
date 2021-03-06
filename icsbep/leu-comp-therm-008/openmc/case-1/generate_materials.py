import openmc

mats = openmc.Materials()

mat = openmc.Material(1)
mat.name = "Water with 1511 PPM"
mat.set_density('sum')
mat.add_nuclide('H1', 6.6737e-02)
mat.add_nuclide('O16', 3.3369e-02)
mat.add_nuclide('B10', 1.6769e-05)
mat.add_nuclide('B11', 6.7497e-05)
mat.add_s_alpha_beta('c_H_in_H2O')
mats.append(mat)

mat = openmc.Material(2)
mat.name = "Fuel (2.459 w/o with B-10 for impurities)"
mat.set_density('sum')
mat.add_nuclide('U234', 4.5689e-06)
mat.add_nuclide('U235', 5.6868e-04)
mat.add_nuclide('U238', 2.2268e-02)
mat.add_nuclide('O16', 4.5683e-02)
mat.add_nuclide('B10', 2.6055e-07)
mats.append(mat)

mat = openmc.Material(3)
mat.name = "Aluminum 6061 Cladding for Fuel"
mat.set_density('sum')
mat.add_element('Mg', 6.2072e-04)
mat.add_element('Al', 5.3985e-02)
mat.add_element('Si', 3.2230e-04)
mat.add_element('Ti', 4.7263e-05)
mat.add_element('Cr', 5.8029e-05)
mat.add_element('Mn', 4.1191e-05)
mat.add_element('Fe', 1.8910e-04)
mat.add_element('Cu', 5.9353e-05)
mat.add_element('Zn', 5.7679e-05)
mats.append(mat)

mats.export_to_xml()
