import numpy as np


class Neuron:
    def __init__(self, input_number):
        """
        W constructor mają być inicjalizowane wszystkie parametry neuronu takie jak:
        - liczba jego wejść
        - wagi wejść (można by je trzemać razem z wejściami. np. oznaczone numerami zamknięte w dict)
        - czy został użyty bias
        - (jeśli tak to jego wartość)
        - szybkość uczenia się
        """
        self.weights = {i: np.random.uniform(-1, 1) for i in range(input_number)}
        pass

    def take_values(self):
        """
        W tej metodzie neuron będzie przyjmował wartości z wejść
        Co więcej musi wywoływać metodę calculate_insput() aby odrazu było to przeliczane
        """
        pass

    def calculate_output(self):
        """
        Tutaj mnożone będą wartośći przekazywane do neuronu przez wagi każdego z wejść.
        Natępnie będą sumowane i dodawany będzie do nich bias
        """
        pass

    def get_output(self):
        """
        Ta funkcja będzie zwracała wynik obliczony przez neuron
        """

    def set_weights(self, new_weights):
        self.weights = new_weights