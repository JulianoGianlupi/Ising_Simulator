import numpy as np
import numpy.random as rng
import os
import matplotlib.pyplot as plt
import seaborn as sbn


class IsingModel:
    """
    Simulates the 2D Ising model with periodic boundaries
    """

    def __init__(self, dim=100, contact_energy=1, temperature=1, max_sim_time=1000, save_loc=r'./', save_frequency=10,
                 save_csv=False, save_npz=True, do_plot_state=False, plot_state_frequency=100, save_energy=False,
                 save_magn=False, e_save_loc=r'energies', m_save_loc=r'magnetizations', calc_energy=False,
                 calc_magn=False):
        # dimension

        self.N = dim
        self.T = temperature
        self.J = contact_energy
        self.mt = max_sim_time
        self.state = self.init_lattice()
        self.mcs = 0
        self.save_loc = os.path.abspath(save_loc)

        if not os.path.isdir(self.save_loc):
            os.makedirs(os.path.abspath(self.save_loc))

        self.e_save_loc = os.path.join(self.save_loc, e_save_loc)
        if not os.path.isdir(self.e_save_loc):
            os.makedirs(self.e_save_loc)

        self.m_save_loc = os.path.join(self.save_loc, m_save_loc)
        if not os.path.isdir(self.m_save_loc):
            os.makedirs(self.m_save_loc)

        self.save_freq = save_frequency
        self.save_csv = save_csv
        self.save_npz = save_npz

        self.calc_energy = calc_energy
        self.calc_magn = calc_magn

        self.save_energy = save_energy
        self.save_magn = save_magn

        if save_energy:
            path = self.save_loc
            path = os.path.abspath(path)
            path = os.path.join(path, 'energy_step_' + str(self.mcs) + '.csv')
            with open(path, 'w') as f:
                pass
            e = self.get_state_energy()
            self.do_save_energy(e)

        if save_magn:
            path = self.save_loc
            path = os.path.abspath(path)
            path = os.path.join(path, 'mag_step_' + str(self.mcs) + '.csv')
            with open(path, 'w') as f:
                pass
            m = self.get_state_magnetization()
            self.do_save_magnetization(m)

        if save_csv:
            self.save_state_as_csv()
        if save_npz:
            self.save_state_as_npz()

        self.do_plot_state = do_plot_state
        if do_plot_state:
            self.plot_state()
        self.plot_state_frequency = plot_state_frequency

        return

    def init_lattice(self):
        return 2 * rng.randint(2, size=(self.N, self.N)) - 1

    def save_state_as_csv(self, file_name='step_'):
        path = self.save_loc
        path = os.path.join(path, file_name + str(self.mcs))
        np.savetxt(path + '.csv', self.state, delimiter=',')

    def save_state_as_npz(self, file_name='step_'):
        path = self.save_loc
        path = os.path.join(path, file_name + str(self.mcs))
        np.savez_compressed(path + '.npz', self.state)

    def do_save_energy(self, e, file_name='energy_step_'):
        path = self.save_loc
        path = os.path.join(path, file_name + str(self.mcs))
        with open(path, 'a') as f:
            f.write('{}, {}'.format(self.mcs, e))

    def do_save_magnetization(self, m, file_name='mag_step_'):
        path = self.save_loc
        path = os.path.join(path, file_name + str(self.mcs))
        with open(path, 'a') as f:
            f.write('{}, {}'.format(self.mcs, m))

    def _do_monte_carlo_step(self):
        stt = self.state
        N = self.N
        J = self.J
        for i in range(N):  # attempt NxN times
            for j in range(N):
                x = rng.randint(0, N)
                y = rng.randint(0, N)

                spin = stt[x, y]
                sum_neig_spins = stt[(x + 1) % N, y] + stt[(x - 1) % N, y] + stt[x, (y + 1) % N] + stt[x, (y - 1) % N]

                dE = 2 * J * spin * sum_neig_spins

                if dE < 0:
                    self.state[x, y] *= -1
                elif rng.rand() < np.exp(-dE / self.T):
                    self.state[x, y] *= -1
        return

    def plot_state(self):
        plt.figure()
        ax = sbn.heatmap(self.state, linewidth=.01)
        plt.title('Time = ' + str(self.mcs) + ' MCS')
        plt.show()

        return

    def do_mcs(self):
        self._do_monte_carlo_step()

    def get_state_magnetization(self):
        return np.sum(self.state)

    def get_state_energy(self):
        N = self.N
        # J = self.J
        stt = self.state
        e = 0
        for x in range(N):
            for y in range(N):
                spin = stt[x, y]
                sum_neig_spins = stt[(x + 1) % N, y] + stt[(x - 1) % N, y] + stt[x, (y + 1) % N] + stt[x, (y - 1) % N]

                e += 2 * spin * sum_neig_spins
        e *= self.J

        return e / 4  # quadruple count correction

    def run(self):
        for t in range(1, self.mt):
            self.mcs = t
            self.do_mcs()

            time_to_save_the_data = not self.mcs % self.save_freq

            if time_to_save_the_data and (self.save_csv or self.save_npz):
                if self.save_csv:
                    self.save_state_as_csv()
                if self.save_npz:
                    self.save_state_as_npz()
            if time_to_save_the_data and self.save_energy:
                e = self.get_state_energy()
                self.do_save_energy(e)

            if time_to_save_the_data and self.save_magn:
                m = self.get_state_magnetization()
                self.do_save_magnetization(m)

            if self.do_plot_state and not self.mcs % self.plot_state_frequency:
                # plt.close()
                self.plot_state()


if __name__ == '__main__':
    IM = IsingModel(save_npz=True, do_plot_state=False, plot_state_frequency=10)
    IM.run()
    a = 5
