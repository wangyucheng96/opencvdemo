

min_record = -8
counter = 0
while True:

    print(counter)
    record = [1, 2, 3]
    if counter % 10 == 0 and counter != 0:
        min_record = min_record + 2
    str_name = "data/record_data " + str(min_record) + ".txt"
    with open(str_name, 'a') as f:
        for i in record:
            f.write(str(i) + ',')
        f.write('\n')
        record.clear()
    counter = counter + 1