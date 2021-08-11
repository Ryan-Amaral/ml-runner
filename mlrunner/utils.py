
"""
Creates a log file with teh specified fields.
"""
def create_log(name, fields):
    with open(name, "w") as f:
        f.write(fields[0])

        [f.write(f",{fields[i]}") 
            for i in range(1, len(fields))]

        f.write("\n")

"""
Fills in a line of the log file.
"""
def update_log(name, values):
    with open(name, "a") as f:
        f.write(str(values[0]))
        
        [f.write(f",{str(values[i])}") 
            for i in range(1, len(values))]

        f.write("\n")