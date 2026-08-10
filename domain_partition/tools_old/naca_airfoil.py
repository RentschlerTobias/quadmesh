import numpy as np
import torch


class NACA_airfoil:
    def __init__(self):

        profils = ['2412', '6412']
        self.naca_profil = profils[np.random.randint(2)]

        scale_range = (0.7, 0.95)

        self.m = int(self.naca_profil[0])/100
        self.p = int(self.naca_profil[1])/10
        self.t = int(self.naca_profil[2:])/100
        self.c = 1.0

        x = np.linspace(0, 1, 100)
        self.random_angle = 0.2 * np.pi * np.random.rand()

        self.suction_side, self.pressure_side = self.naca4(x)

        self.suction_side[-1, :] = self.pressure_side[-1, :]
        self.suction_side[0, :] = self.pressure_side[0, :]

        self.scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        self.suction_side, self.pressure_side = self.scale(
            self.suction_side, self.pressure_side, self.scale_factor)

        self.suction_side_rotated, self.pressure_side_rotated = self.rotate()

    def camber_line(self, x):
        return np.where((x >= 0) & (x <= (self.c * self.p)),
                        self.m * (x / np.power(self.p, 2)) *
                        (2.0 * self.p - (x / self.c)),
                        self.m * ((self.c - x) / np.power(1 - self.p, 2)) * (1.0 + (x / self.c) - 2.0 * self.p))

    def dyc_over_dx(self, x):
        return np.where((x >= 0) & (x <= (self.c * self.p)),
                        ((2.0 * self.m) / np.power(self.p, 2)) *
                        (self.p - x / self.c),
                        ((2.0 * self.m) / np.power(1 - self.p, 2)) * (self.p - x / self.c))

    def thickness(self, x):
        term1 = 0.2969 * (np.sqrt(x/self.c))
        term2 = -0.1260 * (x/self.c)
        term3 = -0.3516 * np.power(x/self.c, 2)
        term4 = 0.2843 * np.power(x/self.c, 3)
        term5 = -0.1015 * np.power(x/self.c, 4)
        return 5 * self.t * self.c * (term1 + term2 + term3 + term4 + term5)

    def naca4(self, x):
        dyc_dx = self.dyc_over_dx(x)
        th = np.arctan(dyc_dx)
        yt = self.thickness(x)
        yc = self.camber_line(x)
        suction_side = np.array([x - yt*np.sin(th), yc + yt*np.cos(th)]).T
        pressure_side = np.array([x + yt*np.sin(th), yc - yt*np.cos(th)]).T

        # Add 0.5 to the y coordinate for normalization (current range y =[-0.5,0.5] transformation => y =[0,1])
        pressure_side[:, 1] = pressure_side[:, 1]+0.5
        suction_side[:, 1] = suction_side[:, 1]+0.5

        return suction_side, pressure_side

    def scale(self, suction_side, pressure_side, scale_factor):
        suction_side_scaled = suction_side * scale_factor
        pressure_side_scaled = pressure_side * scale_factor
        return suction_side_scaled, pressure_side_scaled

    def rotate(self):
        try:
            phi = self.random_angle
            rotation = np.array(
                [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

            center = (np.mean(self.suction_side, axis=0) +
                      np.mean(self.pressure_side, axis=0)) / 2

            translation_to_origin = -center
            translation_back = center

            translated_suction_side = self.suction_side + translation_to_origin
            rotated_suction_side = np.dot(
                translated_suction_side, rotation) + translation_back

            translated_pressure_side = self.pressure_side + translation_to_origin
            rotated_pressure_side = np.dot(
                translated_pressure_side, rotation) + translation_back

            return rotated_suction_side, rotated_pressure_side
        except:
            print('Geometry not generated yet')
