# This file contains some files about writing and parsing csv file
import csv
import os


class CsvManager(object):
    def __init__(self, file_name='test.csv'):
        self.index = []
        self.name = []
        self.quali = []  # quality
        self.compli = []  # complicity
        self.resume_index = 0

        self.file_name = file_name

        self.read()

    def read(self):
        if not os.path.exists(self.file_name):
            print("The CSV file doesn't exist, create one...")
            open(self.file_name, 'w', newline='')

        with open(self.file_name, 'r') as f:
            csv_reader = csv.reader(f)
            count = 0
            for row in csv_reader:
                print('read in...', row[0])
                self.index.append(int(row[0]))  # index
                self.name.append(row[1])  # name
                self.quali.append(int(row[2]))  # quality
                self.compli.append(int(row[3]))  # complicity
                count += 1

            self.resume_index = count
            print('resume from index:', count)

    def write(self):
        with open(self.file_name, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(self.index)):
                csv_writer.writerow([self.index[i], self.name[i], self.quali[i], self.compli[i]])
            print('Write successful, path "', self.file_name, '"')

    def set(self, idx, name, quali, compli):
        if idx < len(self.index):  # rewrite
            self.name[idx] = name
            self.quali[idx] = quali
            self.compli[idx] = compli
        else:  # append
            self.index.append(idx)
            self.name.append(name)
            self.quali.append(quali)
            self.compli.append(compli)

    def __len__(self):
        return len(self.index)


# csv_manager = CsvManager()
# for i in range(10):
#     csv_manager.append(i, 'name.png', 0, 2)
# csv_manager.write()

