from niapy.problems import Problem

#preimenuj
class Clustering(Problem): #ze tu dimenzijo doloc
    def __init__(self, dimension, lower=0, upper=1, *args, **kwargs):
        super().__init__(dimension, lower, upper, *args, **kwargs)

    # Kako dober je solution glede na problem
    def _evaluate(self, x): # vektor 10 stevil 3x4
        scaled_values = ((x - self.lower) / (self.upper - self.lower)) * len(x)

        rounded_values = [round(value) for value in scaled_values]

        rounded_values = [min(max(value, 0), len(x) - 1) for value in rounded_values]

        return sum(value ** 2 for value in rounded_values)


    # 1 korak iz x dobis lokacijen za vse centroide
    # 2 korak za vsako instanco izracunas razdaljo do vseh centroidov
     # 3 korak dodeli instance najblizjemu centroidu
    # 4 korak izracunas kakovost
#minimization manjsi fitnes je bolsi

#helper funckija 2 in 3 korak