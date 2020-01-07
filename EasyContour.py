import numpy as np


class EasyContour:

    def __init__(self, contour_inp):
        if type(contour_inp) != np.ndarray:
            contour_inp = np.array(contour_inp)
        self.cnt_internal = np.squeeze(contour_inp)
        new = []
        for p in self.cnt_internal:
            new.append([p[0], p[1]])
        new = np.array(new, dtype=np.int)
        self.cnt_internal = new
        # self.standard = self.format([[["x", "y"]], [["x", "y"]]], data_type=np.int32)
        self.last_updated = self.cnt_internal
        self.standard = None

    def __str__(self):
        return str(list(self.cnt_internal))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n >= len(self.cnt_internal):
            raise StopIteration
        self.n += 1
        return self.cnt_internal[self.n - 1]

    def __len__(self):
        return len(self.cnt_internal)

    def __getitem__(self, item):
        return self.cnt_internal[item]

    def __setitem__(self, key, value):
        self.cnt_internal[key] = value

    def format(self, format_list, data_type=np.int):
        # Format of format_list is as follows:
        # [["x", "y", 0], ["x", "y", 0]]
        current_level = format_list
        reps = 0
        while reps <= 10 and ("x" not in current_level and "y" not in current_level):
            current_level = current_level[0]
            reps += 1
        if reps >= 10:
            raise ValueError("Invalid format list %s" % format_list)
        layers = []
        for p in range(reps - 1):
            layers = [layers, ]
        x_pos = current_level.index("x")
        y_pos = current_level.index("y")
        formatted = []
        for p in self.cnt_internal:
            item = current_level.copy()
            item[x_pos] = p[0]
            item[y_pos] = p[1]
            for j in range(reps - 1):
                item = [item, ]
            formatted.append(item)
        return np.array(formatted, dtype=data_type)

    def standard_contour(self):
        if self.cnt_internal is not self.last_updated:
            self.last_updated = self.cnt_internal
            self.standard = self.format([[["x", "y"]], [["x", "y"]]], data_type=np.int32)
        return self.standard
