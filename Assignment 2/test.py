import numpy as np
import cmath

# Reading from netlist
import sys

# Checking if file name is provided
if len(sys.argv)==1:
    print('Please provide file name as argument.')
    exit()

# Checking if only one file name is provided
if len(sys.argv)>2:
    print('Please provide only one file name.')
    exit()

filename = sys.argv[1]

try:
    with open(filename) as f:
        lines = f.readlines()

        circuit_started = False     # To ensure only lines between .circuit and .end are read
        l = []
        for line in lines:

            if line[:4] == '.end':
                circuit_started = False

            if circuit_started:
                if '#' in line:
                    line = line[:line.find('#')]    # Ignores comments
                    l.append(line.strip())
                else:
                    l.append(line.strip())

            if line[:8] == '.circuit':
                circuit_started = True

            if line[:3] == '.ac':
                if '#' in line:
                    line = line[:line.find('#')]    # Ignores comments
                    l.append(line.strip())
                else:
                    l.append(line.strip())

    if len(l) == 0 or circuit_started:              # Checks if file is a valid netlist
        print('Please provide a valid netlist.')
        exit()
    
    # Analising each token and extracting information and storing it in a dictionary
    list_of_components = {'R':[], 'L':[], 'C':[], 'V':[], 'I':[], 'E':[], 'G':[], 'H':[], 'F':[]}

    for line in l:
        s = line.split()
        if s[1] == 'GND':
            s[1] = 0
        if s[2] == 'GND':
            s[2] = 0
        if s[0][0] == 'R':
            list_of_components['R'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'value':s[3]})
        if s[0][0] == 'L':
            list_of_components['L'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'value':s[3]})
        if s[0][0] == 'C':
            list_of_components['C'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'value':s[3]})
        if s[0][0] == 'V' and len(s) == 6:
            list_of_components['V'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'type':s[3], 'value':s[4], 'phase':s[5]})
        if s[0][0] == 'V' and len(s) == 5:
            list_of_components['V'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'type':s[3], 'value':s[4], 'phase':0})
        if s[0][0] == 'I' and len(s) == 6:
            list_of_components['I'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'type':s[3], 'value':s[4], 'phase':s[5]})
        if s[0][0] == 'I' and len(s) == 5:
            list_of_components['I'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'type':s[3], 'value':s[4], 'phase':0})
        if s[0][0] == 'E':
            if s[3] == 'GND':
                s[3] = 0
            if s[4] == 'GND':
                s[4] = 0
            list_of_components['E'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'n3':s[3], 'n4':s[4], 'value':s[5]})
        if s[0][0] == 'G':
            if s[3] == 'GND':
                s[3] = 0
            if s[4] == 'GND':
                s[4] = 0
            list_of_components['G'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'n3':s[3], 'n4':s[4], 'value':s[5]})
        if s[0][0] == 'H':
            list_of_components['H'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'vs_name':s[3], 'value':s[4]})
        if s[0][0] == 'F':
            list_of_components['F'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'vs_name':s[3], 'value':s[4]})
        if s[0] == '.ac':
            frequency = float(s[2])  

except FileNotFoundError:
    print('Please enter a valid file name.')
# Find number of nodes

nodes = 0
for k,v in list_of_components.items():
    for component in v:
        if int(component['n1']) > nodes:
            nodes = int(component['n1'])
        if int(component['n2']) > nodes:
            nodes = int(component['n2'])

# Creating Classes and Objects

class Impedance:

    def __init__(self, value, from_node, to_node):
        self.value = value
        self.from_node = from_node
        self.to_node = to_node

class VS:

    def __init__(self, value, from_node, to_node):
        self.value = value
        self.from_node = from_node
        self.to_node = to_node

class CS:

    def __init__(self, value, from_node, to_node):
        self.value = value
        self.from_node = from_node
        self.to_node = to_node

impedances = []
voltage_sources = []
current_sources = []

for r in list_of_components['R']:

    impedances.append(Impedance(float(r['value']), int(r['n1']), int(r['n2'])))

for l in list_of_components['L']:

    impedances.append(Impedance(complex(0, 2*np.pi*frequency*float(l['value'])), int(l['n1']), int(l['n2'])))

for c in list_of_components['C']:

    impedances.append(Impedance(1/complex(0, 2*np.pi*frequency*float(c['value'])), int(c['n1']), int(c['n2'])))

for v in list_of_components['V']:
    
    voltage_sources.append(VS(float(v['value'])*cmath.exp(complex(0, float(v['phase']))), int(v['n1']), int(v['n2'])))

for cs in list_of_components['I']:

    current_sources.append(CS(float(cs['value'])*cmath.exp(complex(0, float(cs['phase']))), int(cs['n1']), int(cs(v['n2']))))


# Setting up the matrix
A = np.zeros((nodes+1, nodes+1), dtype = complex)
b = np.zeros((nodes+1, 1), dtype = complex)
for imp in impedances:

    if imp.from_node != 0 and imp.to_node != 0:
        A[imp.from_node-1, imp.from_node-1]+= 1/imp.value
        A[imp.from_node-1, imp.to_node-1]-= 1/imp.value
        A[imp.to_node-1, imp.from_node-1]-= 1/imp.value
        A[imp.to_node-1, imp.to_node-1]+= 1/imp.value

    elif imp.from_node == 0 and imp.to_node != 0:
        A[imp.to_node-1, imp.to_node-1] += 1/imp.value

    elif imp.from_node != 0 and imp.to_node == 0:
        A[imp.from_node-1, imp.from_node-1] += 1/imp.value

    else:
        print('Invalid Circuit')
        exit()

for v in voltage_sources:

    if v.from_node != 0 and v.to_node != 0:
        A[-1, v.to_node-1] = 1
        A[-1, v.from_node-1] = -1
        b[-1, 0] = v.value
        A[v.to_node-1,-1]-=1
        A[v.from_node-1,-1]+=1

    elif v.from_node == 0 and v.to_node != 0:
        A[-1, v.to_node-1] = 1
        b[-1, 0] = v.value
        A[v.to_node-1,-1]-=1

    elif v.from_node != 0 and v.to_node == 0:
        A[-1, v.from_node-1] = -1
        b[-1, 0] = v.value
        A[v.from_node-1,-1]+=1

    else:
        print('Invalid Circuit')
        exit()

for curr in current_sources:

    if curr.from_node != 0 and curr.to_node != 0:
        b[curr.from_node-1]-=curr.value
        b[curr.to_node-1]+=curr.value

    elif curr.from_node == 0 and curr.to_node != 0:
        b[curr.to_node-1]+=curr.value

    elif curr.from_node != 0 and curr.to_node == 0:
        b[curr.from_node-1]-=curr.value

    else:
        print('Invalid Circuit')
        exit()

# Solving

x = np.linalg.solve(A,b.T.squeeze())

for i in range(nodes):
    print(f'Voltage at node {i+1} = {x[i]}')
    
print(f'Current through Voltage Source = {x[-1]}')