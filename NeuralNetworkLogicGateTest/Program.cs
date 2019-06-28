using System;
using System.Collections.Generic;

namespace NeuralNetworkLogicGateTest
{
    class Program
    {
        static Random rand = new Random();
        private static int train(int trainToo, int epoch, bool randomTrain, NeuralNetwork NN, double[][] inputs, double[][] expectedOutputs, double learningRate)
        {
            int timesTrained = 0;
            int next = -1;
            while (timesTrained / epoch < trainToo)
            {
                if (randomTrain)
                {
                    next = rand.Next(inputs.Length);
                }
                else
                {
                    next++;
                    if (next > inputs.Length - 1)
                        next = 0;
                }

                NN.FeedForward(inputs[next]);
                NN.backpropagation(expectedOutputs[next], learningRate, epoch);
                timesTrained++;
            }
            return timesTrained;
        }

        static void Main(string[] args)
        {
            int numOfInputs = 2;
            int[] NNArchitecture = new int[] { 1,4,2};
            NeuralNetwork NN = new NeuralNetwork(NNArchitecture, numOfInputs);

            bool exit = false;
            ConsoleKeyInfo ckinput;
            int timesTrained = 0;
            int epoch = 1;
            int trainToo = 6000;
            double learningRate = 0.01;
            bool randomTrain = false;

            double[][] inputs = new double[][] {
                //Typical inputs
                //new double[2] { 0, 0 },
                //new double[2] { 0, 1 },
                //new double[2] { 1, 0 },
                //new double[2] { 1, 1 }
                //Normalized Inputs
                new double[2] { -1, -1 },
                new double[2] { -1, 1 },
                new double[2] { 1, -1 },
                new double[2] { 1, 1 }
                //Binary Inputs
                //new double[1] { 1 },
                //new double[1] { 0 }
            };

            double[][] expectedOutputs =
            //AND
            new double[][] {
                new double[] { 0.01 },
                new double[] { 0.01 },
                new double[] { 0.01 },
                new double[] { 0.99 }
            };
            //AND, NAND
            //new double[][] {
            //    new double[] { 0, 1},
            //    new double[] { 0, 1},
            //    new double[] { 0, 1},
            //    new double[] { 1, 0}
            //};
            //OR, NOR
            //new double[][] {
            //    new double[] { 0, 1 },
            //    new double[] { 1, 0 },
            //    new double[] { 1, 0 },
            //    new double[] { 1, 0 }
            //};
            //Spit out the same output as the input
            //new double[][] {
            //    new double[] { 1 },
            //    new double[] { 0 }
            //};

            timesTrained = train(trainToo, epoch, randomTrain, NN, inputs, expectedOutputs, learningRate);

            do
            {
                Console.WriteLine("This is the Neural Network Logic Gate Test\n");
                for (int i = 0; i < inputs.Length; i++)
                {
                    //int next = rand.Next(4);
                    NN.FeedForward(inputs[i]);
                    NN.backpropagation(expectedOutputs[i], learningRate, epoch);
                    Console.Write("Training with inputs: ");
                    Console.WriteLine("[{0}]", string.Join(", ", inputs[i]));
                    Console.Write("Expected Output: ");
                    Console.WriteLine("[{0}]", string.Join(", ", expectedOutputs[i]));
                    Console.Write("Output: ");
                    Console.WriteLine("[{0}]", string.Join(", ", NN.getOutputs()));
                    Console.WriteLine("Total Error: {0}\n", NN.RecentTotalError);
                    timesTrained++;
                }
                Console.WriteLine("Systems was trained {0} times.", timesTrained / epoch);

                Console.WriteLine("Hit x to exit");
                Console.WriteLine("Hit r to reset the network\nPress any other key to simulate another epoch");
                ckinput = Console.ReadKey();
                Console.Clear();
                if (ckinput.Key == ConsoleKey.X)
                    exit = true;
                else if (ckinput.Key == ConsoleKey.R)
                {
                    //NN = new NeuralNetwork(numOfInputs, numOfHiddenLayers, numOfNeuronsPerHiddenLayer, numOfOutputs);
                    NN = new NeuralNetwork(NNArchitecture, numOfInputs);
                    timesTrained = train(trainToo, epoch, randomTrain, NN, inputs, expectedOutputs, learningRate);
                }
            }
            while (!exit);


        }

    }
}

class NeuralNetwork
{
    static Random rand = new Random();
    private int train;
    private int epoch;
    private double mostRecentTotalError;
    private double[] inputLayer;
    private Neuron[][] hiddenLayer;
    private Neuron[] outputLayer;

    public int NumOfTraining { get { return train; } }
    public int NumOfEpoch { get { return epoch; } }
    public double RecentTotalError { get { return mostRecentTotalError; } }

    //public NeuralNetwork(int numberOfInputs, int hiddenLayerNumber, int neuronsPerHiddenLayer, int outputNeurons)
    //{
    /// <summary>
    /// Create a Neural Network.
    /// </summary>
    /// <param name="NetworkSetup">index 0 is the number of output neurons, all other indexes are hidden layer neurons</param>
    /// <param name="numberOfInputs">The number of inputs</param>
    public NeuralNetwork(int[] NetworkSetup, int numberOfInputs)
    {
        inputLayer = new double[numberOfInputs];
        if (NetworkSetup.Length < 1 || NetworkSetup[1] == 0)
            throw new Exception("NeuralNetwork: Must have neurons in the hidden layer.");
        outputLayer = new Neuron[NetworkSetup[0]];
        hiddenLayer = new Neuron[NetworkSetup.Length - 1][];
        for (int i = 0; i < NetworkSetup.Length; i++)
        {
            if (i != 0)
                hiddenLayer[i - 1] = new Neuron[NetworkSetup[i]];
            for (int j = 0; j < NetworkSetup[i]; j++)
            {
                if(i == 0)
                    outputLayer[j] = new Neuron(NetworkSetup[i + 1], rand, 0, "OL[" + j.ToString() + "]");
                else if(i != NetworkSetup.Length - 1)
                    hiddenLayer[i - 1][j] = new Neuron(NetworkSetup[i + 1], rand, 0, "HL[" + i.ToString() + "][" + j.ToString() + "]");
                else
                    hiddenLayer[i - 1][j] = new Neuron(numberOfInputs, rand, 0, "IL[" + i.ToString() + "][" + j.ToString() + "]");
            }
        }
        train = 0;
        epoch = 0;
    }

    //These two constructors is to make the Neural Network serializable.
    private NeuralNetwork(int trn, int epch, double[] inlayer, Neuron[][] HL, Neuron[] outLayer)
    {
        train = trn;
        epoch = epch;
        inputLayer = inlayer;
        hiddenLayer = HL;
        outputLayer = outLayer;
    }
    private NeuralNetwork()
    { }

    /// <summary>
    /// Returns the values of the output layer.  Should be used after FeedForward().
    /// </summary>
    /// <returns></returns>
    public double[] getOutputs()
    {
        double[] retVal = new double[outputLayer.Length];
        for (int i = 0; i < outputLayer.Length; i++)
        {
            retVal[i] = outputLayer[i].getOutput;
        }
        return retVal;
    }

    /// <summary>
    /// Calculates the output for each layer of the Neural Network.  After calling this function, use getOutputs() to collect the output of the network.
    /// </summary>
    /// <param name="inputs">Inputs for the Neural Network.</param>
    public void FeedForward(double[] inputs)
    {
        if (inputs.Length != inputLayer.Length)
            throw new Exception("FeedForward: Incorrect number of inputs submitted.");
        inputLayer = inputs;

        for (int i = hiddenLayer.Length - 1; i >= 0; i--)
        {
            for (int j = 0; j < hiddenLayer[i].Length; j++)
            {
                if (i == hiddenLayer.Length - 1)
                    hiddenLayer[i][j].CalcNewOuput(inputLayer);
                else
                {
                    double[] prevLayerOutput = getLayerOutput(i + 1);
                    hiddenLayer[i][j].CalcNewOuput(prevLayerOutput);
                }
            }
        }

        for (int i = 0; i < outputLayer.Length; i++)
        {
            outputLayer[i].CalcNewOuput(getLayerOutput(0));
        }
    }

    /// <summary>
    /// Stores the outputs of a hidden layer into a double array. 
    /// </summary>
    /// <param name="layer">The layer you'd like the outputs of.</param>
    /// <returns></returns>
    private double[] getLayerOutput(int layer)
    {
        int size = hiddenLayer[layer].Length;
        double[] retVal = new double[size];
        for (int i = 0; i < size; i++)
        {
            retVal[i] = hiddenLayer[layer][i].getOutput;
        }
        return retVal;
    }

    private double inverseSigmoid(double x)
    {
        //double d = sigmoidFunction(x);
        return x * (1 - x);
    }

    /// <summary>
    /// Backpropagation so the Neural Network can learn.
    /// </summary>
    /// <param name="expectedOutputs">A double array of the expected outputs</param>
    /// <param name="learningRate">How quickly the network can learn.</param>
    /// <param name="trainForEpochNum">How many training operations performed before weights are updated.</param>
    public void backpropagation(double[] expectedOutputs, double learningRate, int trainForEpochNum)
    {
        if (outputLayer.Length != expectedOutputs.Length)
            throw new Exception("train: Output layer and expected layer not equal.");
        double TotalError = 0;

        //Calculate weight update of output neurons.
        for (int i = 0; i < outputLayer.Length; i++)
        {
            TotalError += Math.Pow((expectedOutputs[i] - outputLayer[i].getOutput), 2);
            for (int j = 0; j < hiddenLayer[0].Length; j++)
            {
                // Etotal/w = -(target - out) * out(1 - out) * outhln
                outputLayer[i].addTrainingOperation(j, -(expectedOutputs[i] - outputLayer[i].getOutput) * inverseSigmoid(outputLayer[i].getOutput) * hiddenLayer[0][j].getOutput);
            }
        }
        //Store total error.
        mostRecentTotalError = TotalError;

        //Calculate weight update for hidden layers.
        for (int i = 0; i < hiddenLayer.Length - 1; i++)
        {
            //Calculates the total error for the layer
            massCalcOutputEffectOnError(i, expectedOutputs);
            for (int j = 0; j < hiddenLayer[i].Length; j++)
            {
                double error = hiddenLayer[i][j].OutputEffectOnTotalError * inverseSigmoid(hiddenLayer[i][j].getOutput);
                for (int k = 0; k < hiddenLayer[i][j].getWeightCount; k++)
                {
                    //Etotal/wn = Sum of total error * hout(1 - hout) * h+1out
                    hiddenLayer[i][j].addTrainingOperation(k, error * hiddenLayer[i + 1][k].getOutput);
                }

            }

        }

        //Calculate weight update for input hidden layer
        int il = hiddenLayer.Length - 1;
        massCalcOutputEffectOnError(il, expectedOutputs);
        for (int j = 0; j < hiddenLayer[il].Length; j++)
        {
            double error = hiddenLayer[il][j].OutputEffectOnTotalError * inverseSigmoid(hiddenLayer[il][j].getOutput);
            for (int k = 0; k < inputLayer.Length; k++)
            {
                //Etotal/wn = Sum of total error * hout(1 - hout) * h+1out
                hiddenLayer[il][j].addTrainingOperation(k, error * inputLayer[k]);
            }
        }

        train++;
        //update weights if we've reached an epoch.
        if (train % trainForEpochNum == 0)
        {
            epoch++;
            foreach (Neuron N in outputLayer)
                N.UpdateWeights(learningRate);
            foreach (Neuron[] Layer in hiddenLayer)
                foreach (Neuron N in Layer)
                {
                    N.UpdateWeights(learningRate);
                }
        }
    }

    /// <summary>
    /// Calculates the total effect on error of each neuron in a layer.  Error is stored in the neuron.
    /// </summary>
    /// <param name="hiddenLayerIndex">Which hidden layer to calculate the effect on total error.  Should go from 0 -> n. </param>
    /// <param name="expectedOutputs">Expected outputs for this training operation.</param>
    private void massCalcOutputEffectOnError(int hiddenLayerIndex, double[] expectedOutputs)
    {
        double sum = 0;
        if (hiddenLayerIndex == 0)
        {
            for (int j = 0; j < hiddenLayer[hiddenLayerIndex].Length; j++)
            {
                sum = 0;
                for (int i = 0; i < outputLayer.Length; i++)
                {
                    //sum = -(target - outi) * outi(1 - outi) * HLi weightj
                    sum += -(expectedOutputs[i] - outputLayer[i].getOutput) * (outputLayer[i].getOutput * (1 - outputLayer[i].getOutput)) * outputLayer[i].getWeight(j);
                }
                hiddenLayer[hiddenLayerIndex][j].OutputEffectOnTotalError = sum;
            }
        }
        else
        {
            for (int i = 0; i < hiddenLayer[hiddenLayerIndex].Length; i++)
            {
                sum = 0;
                for (int j = 0; j < hiddenLayer[hiddenLayerIndex - 1].Length; j++)
                {
                    //sum = HLj-1 effectOnTotalError * HLj-1 weighti
                    sum += hiddenLayer[hiddenLayerIndex - 1][j].OutputEffectOnTotalError * hiddenLayer[hiddenLayerIndex - 1][j].getWeight(i);
                }
                hiddenLayer[hiddenLayerIndex][i].OutputEffectOnTotalError = sum;
            }
        }
    }
}
class Neuron
{
    protected string neuronName;
    protected double[] weight;
    protected double output;
    protected double bias;
    protected double[] updateValues;
    protected int[] updateValuesCounter;
    protected double outputEffectOnTotalError;


    public double getOutput { get { return output; } }
    public double getWeight(int i) { return weight[i]; }
    public int getWeightCount { get { return weight.Length; } }
    public double OutputEffectOnTotalError { get { return outputEffectOnTotalError; } set { this.outputEffectOnTotalError = value; } }
    public string NeuronName { get { return neuronName; } }

    public Neuron(int weightNum, Random rand, double neuronBias, string Name)
    {
        weight = new double[weightNum];
        for (int i = 0; i < weightNum; i++)
        {
            weight[i] = (rand.NextDouble() * 2) - 1;
        }
        output = 0;
        bias = neuronBias;
        updateValues = new double[weightNum];
        updateValuesCounter = new int[weightNum];
        outputEffectOnTotalError = 0;
        neuronName = Name;
    }

    public void CalcNewOuput(double[] inputs)
    {
        double sum = 0;
        if (inputs.Length != weight.Length)
            throw new Exception("CalcNewOutput: Incorrect number of inputs entered into neuron.");
        for (int i = 0; i < inputs.Length; i++)
        {
            sum += inputs[i] * weight[i];
        }

        sum += bias;
        output = sigmoidFunction(sum);
        if (Double.IsNaN(output))
            throw new Exception("CalcNewOuput: Output is NaN, I don't know why.");
    }

    private double sigmoidFunction(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public void adjustWeightsAndBias(double error, double[] inputs)
    {
        if (inputs.Length != weight.Length)
            throw new Exception("adjustWeightsAndBias: Incorrect number of inputs compared to weights of neuron.");
        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] += inputs[i];
        }
        bias += error;
    }

    public void addTrainingOperation(int weightIndex, double error)
    {
        if (Double.IsNaN(error))
            throw new Exception("addTrainingOperation: error value is NaN.");
        updateValues[weightIndex] += error;
        updateValuesCounter[weightIndex]++;
    }

    public void UpdateWeights(double learningRate)
    {
        //This switch block is for easy dubugging.
        //switch (this.neuronName[0])
        //{
        //    case 'I':
        //        break;
        //    case 'H':
        //        break;
        //    case 'O':
        //        break;
        //}

        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = weight[i] - (learningRate * (updateValues[i] / updateValuesCounter[i]));

            if (Double.IsNaN(weight[i]))
                throw new Exception("CalcNewOuput: Output is NaN, I don't know why.");
            updateValuesCounter[i] = 0;
        }
    }

}