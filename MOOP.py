import numpy as np # The library numpy will be used for data management
from customer import Customer
from port import Port
from collections import Counter #auxiliar
import itertools #auxiliar
from platypus import NSGAII, Problem, Real, Integer #The library platypus will be used for MOOP with genetic algorithms


class Moop:
    def __init__(self, port, customers, customersOutput):
        # load required variables from port and customers
        self.port = port
        self.customers = customers
        self.customersOutput = customersOutput
        self.containerStack = port.containerStack
        self.containerInfo = port.containerInfo
        self.currentStack = port.containerStack.copy()
        self.remainingMoves = 150
        self.moves = np.zeros((10, 10))

        self.promoted2updatedPrice = {}
        self.exchanged = []
        for output in customersOutput:
            with open(output, 'r') as f:
                f.readline()
                for line in f:
                    line = line.strip().replace(" ", "")
                    data = line.split(",")
                    promoted = int(data[0])
                    updatedPrice = float(data[1])
                    assert promoted not in self.promoted2updatedPrice
                    self.promoted2updatedPrice[promoted] = updatedPrice
                    removed = data[2:]
                    removed[0] = removed[0][1:]
                    removed[-1] = removed[-1][:-1]
                    removed = [int(s[1:-1]) for s in removed]
                    self.exchanged += removed

    def removeCompulsory(self):
        # REMOVE COMPULSORY CONTAINERS AND THE ONES ABOVE THEM THAT HAVE TO BE REMOVED
        # IN ORDER TO REMOVE THE COMPULSORY ONES

        self.compulsoryContainers = []
        for customer in self.customers:
            for container in customer.includedByDefault:
                if int(container) not in self.exchanged:
                    self.compulsoryContainers.append(int(container))

        for x in range(self.containerStack.shape[0]):
            for y in range(self.containerStack.shape[1]):
                z = 0
                container = self.containerStack[x, y, z]
                while container not in self.compulsoryContainers and z < 5:
                    z += 1
                    if z < 5:
                        container = self.containerStack[x, y, z]
                    else:
                        container = None
                if z < 5:
                    while container != 0 and z < 5:
                        self.currentStack[x, y, z] = 0
                        self.moves[x, y] += 1
                        self.remainingMoves -= 1
                        z += 1
                        if z < 5:
                            container = self.containerStack[x, y, z]
                        else:
                            container = 0

    def computeProfitAndUpdatedContractPrices(self):
        # COMPUTE THE PROFIT FOR EACH OF THE REMAINING CONTAINERS WITH THE
        # UPDATED CONTRACT PRICES, AND THE UPDATED CONTRACT PRICES AS WELL

        self.updatedContractPrices = np.zeros(self.containerStack.shape)
        self.profit = np.zeros((*self.containerStack.shape, 3))

        for container in self.containerInfo:
            x, y, z = self.containerInfo[container][-3:]
            if container in self.promoted2updatedPrice:
                contractPrice = self.promoted2updatedPrice[container]
                assert contractPrice >= self.containerInfo[container][0]
            else:
                contractPrice = self.containerInfo[container][0]
            bussinesValue = self.containerInfo[container][1]
            profit = bussinesValue - contractPrice
            self.updatedContractPrices[x, y, z] = contractPrice
            if self.containerInfo[container][2] == "Carrefour":
                self.profit[x, y, z, :] = [profit, 0, contractPrice]
            elif self.containerInfo[container][2] == "Metro":
                self.profit[x, y, z, :] = [0, profit, contractPrice]
            else:
                self.profit[x, y, z, :] = [0, 0, contractPrice]

    def computePossibleMoves(self):
        # FOR EACH STACK ((x, y) coordinate) COMPUTE HOW MANY CONTAINERS CAN BE
        # REMOVED AND FOR EACH POSSIBLE AMOUNT OF REMOVALS COMPUTE THE ADDED
        # PROFIT FOR EACH OF THE CUSTOMERS, THE ADDED CONTRACT PRICES AND THE
        # AMOUNT (the c + 2 objectives of the pareto front)
        amountOfPossibleMoves = 0
        self.possibleMoves = [[[] for y in range(self.currentStack.shape[1])] for x in range(self.currentStack.shape[0])]
        for x, y in itertools.product(range(self.currentStack.shape[0]),
                                      range(self.currentStack.shape[0])):
            self.possibleMoves[x][y] = []
            z = 4
            while z >= 0 and self.currentStack[x, y, z] == 0:
                # print(self.currentStack[x, y, z])
                z -= 1

            sizeOfStack = z + 1
            amountOfPossibleMoves += sizeOfStack + 1
            requiredMoves = 1
            while z >= 0:
                profits = self.profit[x, y, z, :]
                profitsPerMove = np.sum(self.profit[x, y, z:sizeOfStack, :], axis=0)/requiredMoves
                self.possibleMoves[x][y].append(np.array((*profitsPerMove, requiredMoves)))
                z -= 1
                requiredMoves += 1

    def computeResult(self, solution):
        """
        Given a solution matrix returns the values of each objective
        """
        result = np.array([0., 0., 0., 0.])
        for x, y in itertools.product(range(self.currentStack.shape[0]),
                                      range(self.currentStack.shape[0])):
            numberOfContainersRemoved = solution[x][y]
            if numberOfContainersRemoved != 0:
                possibleMoves = self.possibleMoves[x][y]
                value = possibleMoves[numberOfContainersRemoved-1]
                value[-1] = 1
                result += value*numberOfContainersRemoved
        return result

    def optimize(self):
        """
        Finds the optimized solution for the problem using multi objective
        genetic algorithms to build a pareto front
        """

        # 1
        self.removeCompulsory()

        # 2
        self.computeProfitAndUpdatedContractPrices()

        # 3
        self.computePossibleMoves()

        self.paretoFront = []
        # A PARETO SOLUTION MAKES LESS THAN 150 MOVES AND NO OTHER SOLUTION
        # GETS BETTER VALUES FOR ALL C+2 OBJECTIVES (neither some equal and at
        # least one better)

        def convertVariables2Matrix(x):
            """
            Converts problem variables to a solution matrix
            """
            xy = list(itertools.product(range(self.currentStack.shape[0]), range(self.currentStack.shape[1])))
            variable = [1 if len(self.possibleMoves[x][y]) > 0 else 0 for (x,y) in xy]
            a = []
            i = 0
            for j in range(len(list(variable))):
                if variable[j] == 1:
                    a.append(x[i])
                    i += 1
                else:
                    a.append(0)
            a = np.array(a)
            a = np.reshape(a, [10, 10])
            return a

        def func(x):
            """
            function that evaluates all the objectives of the solution and the
            restriction
            """
            a = convertVariables2Matrix(x)
            result = self.computeResult(a)
            constraint = result[-1]-self.remainingMoves
            return result, [constraint]

        # set of all x, y combinations
        xy = list(itertools.product(range(self.currentStack.shape[0]), range(self.currentStack.shape[1])))

        # the variables of the problem, integers representing how many containers
        # will be deliver from a given (x, y) coordinate
        integers = [Integer(0, len(self.possibleMoves[x][y])) for (x, y) in xy if len(self.possibleMoves[x][y]) > 0]

        # problem definition in platypus
        problem = Problem(len(integers), 4, 1)
        problem.types[:] = integers
        problem.function = func
        problem.directions[:] = Problem.MAXIMIZE
        problem.constraints[:] = "<=0"

        # algorithm choice (NSGAII)
        algorithm = NSGAII(problem)
        algorithm.run(1000000)

        # extracting results from the algorithm
        paretoFront = []
        for solution in algorithm.result:
            if solution.feasible:
                variables = []
                for i in range(len(integers)):
                    variables.append(integers[i].decode(solution.variables[i]))
                a = convertVariables2Matrix(variables)
                paretoFront.append((a, list(solution.objectives)))

        # finding closest solution to average result
        results = np.array([solution[1] for solution in paretoFront])
        averageResult = results.mean(axis=0)
        distances2Mean = []
        for result in results:
            distance2Mean = [((result[i]-averageResult[i])/averageResult)**2 for i in range(len(averageResult))]
            distance2Mean = np.sum(distance2Mean)
            distances2Mean.append(distance2Mean)
        bestIndex = np.argmin(distances2Mean)
        finalSolution = paretoFront[bestIndex][0] + self.moves

        # generating final report
        totalContractPrice = 0
        totalProfits = [0, 0]
        totalDelivered = 0
        movesDetails = []
        for x, y in itertools.product(range(finalSolution.shape[0]), range(finalSolution.shape[1])):
            amount2deliver = finalSolution[x][y]
            topContainer = self.port.getTopContainer(x, y)
            z = topContainer
            while amount2deliver > 0:
                containerID = self.containerStack[x, y, z]

                if containerID in self.promoted2updatedPrice:
                    contractPrice = self.promoted2updatedPrice[containerID]
                    assert contractPrice >= self.containerInfo[containerID][0]
                else:
                    contractPrice = self.containerInfo[containerID][0]
                bussinesValue = self.containerInfo[containerID][1]
                profit = bussinesValue - contractPrice
                totalContractPrice += contractPrice
                totalDelivered += 1
                if self.containerInfo[containerID][2] == "Carrefour":
                    totalProfits[0] += profit
                elif self.containerInfo[containerID][2] == "Metro":
                    totalProfits[1] += profit
                # if containerID in self.promoted2updatedPrice:
                movesDetails.append(f"Deliver container {containerID} from coordinate ({x+1}, {y+1}, {z+1}) with updated price {contractPrice}")
                # else:
                #     movesDetails.append(f"Deliver container {containerID} from coordinate ({x+1}, {y+1}, {z+1})")
                amount2deliver -= 1
                z -= 1
        with open("Final report.txt", 'w') as f:
            for line in movesDetails:
                f.write(line+"\n")
        with open("Final report performance.txt", 'w') as f:
            f.write(f"Total contract prices = {totalContractPrice}\n")
            f.write(f"Carrefour profits = {totalProfits[0]}\n")
            f.write(f"Metro profits = {totalProfits[1]}\n")
            f.write(f"Port Throughput = {totalDelivered}\n")
        print(totalContractPrice, totalProfits, totalDelivered)


if __name__ == "__main__":
    carrefour = Customer("carrefour_report.csv", "container.csv",
                         name="carrefour")
    metro = Customer("metro_report.csv", "container.csv", name='metro')
    port = Port("container.csv")
    moop = Moop(port,
                [carrefour, metro],
                ["carrefour_output.csv", "metro_output.csv"])

    moop.optimize()
