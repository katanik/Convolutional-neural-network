package com.company.neural_network_with_backpropogation;


import com.company.neural_network.Layer;
import com.company.neural_network.Network;
import com.company.neural_network.Size;

public class BackpropogationNetwork extends Network {
    public BackpropogationNetwork (){
        super();
    }

    public BackpropogationNetwork(Layer[] layers) {
        super(layers);
        randomize(0, 0.00000003);
    }

    public void randomize(double min, double max) {
        final int size = getSize();
        for (int i = 0; i < size; i++) {
            Layer layer = getLayer(i);
            if (layer instanceof BackpropogationLayer) ((BackpropogationLayer) layer).randomize(min, max);
        }
    }

    public double[][][] computeOutput(double[][][] input){
        if (input==null || !getInputSize().compare(new Size(input)))
            throw new IllegalArgumentException();

        final int size = getSize();
        double[][][][] outputs = new double[size][][][];
        outputs[0] = getLayer(0).computeOutput(input);
        for (int i = 1; i < size; i++) {
            outputs[i] = getLayer(i).computeOutput(outputs[i - 1]);
        }
        return outputs[size-1];
    }

    public double learnPattern(double[][][] input, double[][][] goal, double rate, double momentum) {
        if (input == null || !getInputSize().compare(new Size(input)) || goal == null || !getOutputSize().compare(new Size(goal)))
            throw new IllegalArgumentException();

        // do pass forward
        final int size = getSize();
        double[][][][] outputs = new double[size][][][];
        outputs[0] = getLayer(0).computeOutput(input);
        for (int i = 1; i < size; i++) {
            outputs[i] = getLayer(i).computeOutput(outputs[i - 1]);
        }

        // compute error of output layer
        Layer layer = getLayer(size - 1);
        final Size layerSize = layer.getSize();
        double[][][] error = new double[layerSize.depth][layerSize.width][layerSize.height];
        double totalError = 0;

        for (int i = 0; i < layerSize.depth; i++) {
            for (int x=0; x<layerSize.width; x++){
                for (int y=0; y< layerSize.height; y++) {
                    error[i][x][y]=goal[i][x][y]-outputs[size-1][i][x][y];
                    totalError += Math.abs(goal[i][x][y]-outputs[size-1][i][x][y]);
                }
            }
        }

        // update output layer
        if (layer instanceof BackpropogationLayer)
            ((BackpropogationLayer) layer).adjust(size == 1 ? input : outputs[size - 2], error, rate, momentum);

        // go on hidden layers
        double[][][] previousError = error;
        Layer previousLayer = layer;

        for (int i = size - 2; i >= 0; i--, previousError = error, previousLayer = layer) {
            // get next layer
            layer = getLayer(i);

            // compute error for next layer
            if (previousLayer instanceof BackpropogationLayer)
                error=((BackpropogationLayer)previousLayer).computeBackwardError(outputs[i], previousError);
            else
                error=previousError;

            // update layer
            if (layer instanceof BackpropogationLayer)
                ((BackpropogationLayer)layer).adjust(i==0?input:outputs[i-1], error, rate, momentum);
        }
        // return total error
        return totalError;
    }

}
