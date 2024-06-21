def read_data_ucl (filename):

    data = []
    pi_values = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            tmp = line.split('|')
            if len(tmp)>1:
                teams = tmp[2].split(',')
                for i in range(0, len(teams)):
                    pi_values[teams[i]] = 1.0
                data.append(teams)


    return data, pi_values