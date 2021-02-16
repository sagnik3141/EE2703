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

    if len(l) == 0 or circuit_started:              # Checks if file is a valid netlist
        print('Please provide a valid netlist.')
        exit()
    
    # Analising each token and extracting information and storing it in a dictionary
    list_of_components = {'R':[], 'L':[], 'C':[], 'V':[], 'I':[], 'E':[], 'G':[], 'H':[], 'F':[]}

    for line in l:
        s = line.split()
        if s[0][0] == 'R':
            list_of_components['R'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'value':s[3]})
        if s[0][0] == 'L':
            list_of_components['L'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'value':s[3]})
        if s[0][0] == 'C':
            list_of_components['C'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'value':s[3]})
        if s[0][0] == 'V':
            list_of_components['V'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'value':s[3]})
        if s[0][0] == 'I':
            list_of_components['I'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'value':s[3]})
        if s[0][0] == 'E':
            list_of_components['E'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'n3':s[3], 'n4':s[4], 'value':s[5]})
        if s[0][0] == 'G':
            list_of_components['G'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'n3':s[3], 'n4':s[4], 'value':s[5]})
        if s[0][0] == 'H':
            list_of_components['H'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'vs_name':s[3], 'value':s[4]})
        if s[0][0] == 'F':
            list_of_components['F'].append({'name':s[0], 'n1':s[1], 'n2':s[2], 'vs_name':s[3], 'value':s[4]})  

    # Reversing each token
    for i in range(len(l)):
        l[i] = l[i].split()
        l[i].reverse()

    # Printing the reversed tokens in reverse order
    for i in range(len(l)):
        print(' '.join(l[-i-1]))

except FileNotFoundError:
    print('Please enter a valid file name.')