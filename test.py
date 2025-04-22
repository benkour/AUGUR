# %%
import json
import periodictable
from mendeleev import element
# %%


def get_element_properties(element_symbols):
    element_properties = {}

    for symbol in element_symbols:
        # Get atomic number and mass from periodictable
        el = periodictable.elements.symbol(symbol)
        atomic_number = el.number
        atomic_mass = el.mass

        # Get atomic radius and electronegativity from mendeleev
        e = element(el.number)
        atomic_radius = e.atomic_radius if e.atomic_radius is not None else 'N/A'
        electronegativity = e.en_pauling if e.en_pauling is not None else 'N/A'

        # Create a dictionary for the element
        element_properties[symbol] = {
            'atomic_number': atomic_number,
            'atomic_mass': atomic_mass,
            'atomic_radius': atomic_radius,
            'electronegativity': electronegativity
        }

    return element_properties


# Example usage
element_symbols = ["H", "C", "C", "O"]
properties = get_element_properties(element_symbols)

print(properties)

# %%


# Create dictionaries for atomic number, electronegativity, atomic radius, and atomic mass
atom_to_num = {}
atom_to_en = {}
atom_to_r = {}
atom_to_mass = {}

# Populate the dictionaries
for el in range(1, 119):  # Elements from 1 (Hydrogen) to 118 (Oganesson)
    e = element(el)
    atom_to_num[e.symbol] = e.atomic_number
    # Pauling electronegativity
    atom_to_en[e.symbol] = e.en_pauling if e.en_pauling is not None else 'N/A'
    # Atomic radius
    atom_to_r[e.symbol] = e.atomic_radius if e.atomic_radius is not None else 'N/A'
    # Atomic mass
    atom_to_mass[e.symbol] = e.atomic_weight if e.atomic_weight is not None else 'N/A'

# Combine the dictionaries into one
data = {
    "atom_to_num": atom_to_num,
    "atom_to_en": atom_to_en,
    "atom_to_r": atom_to_r,
    "atom_to_mass": atom_to_mass
}

# Save the dictionaries to a JSON file
with open('element_properties.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Dictionaries saved to element_properties.json")

# %%
with open('element_properties.json', 'r') as json_file:
    data = json.load(json_file)
    atom_to_num = data['atom_to_num']
    atom_to_en = data['atom_to_en']
    atom_to_r = data['atom_to_r']
    atom_to_mass = data['atom_to_mass']
# %%
# check if string exists in this list ["C", "H", "O", "N"]
string = "C"
if string in ["C", "H", "O", "N"]:
    print("String exists in the list")

# %%
