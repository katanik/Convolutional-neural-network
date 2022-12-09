package com.company.neural_network;

import java.io.Serializable;

/**
 *Neural layer interface
 */
public interface Layer extends Serializable{

    /**
     * Get the size of input vector
     * @return size of input layer
     */
    Size getInputSize();

    /**
     * Get the size of the layer
     * @return size of layer
     */
    Size getSize();

    /**
     * Compute the response of the layer
     * @param input input vector
     * @return output vector
     */
    double[][][] computeOutput (double[][][] input);
}
