package com.company.neural_network;

import java.io.*;

public class Network implements Serializable{
    private Layer[] layers;

    public Network (Layer[] layers){
        if (layers==null || layers.length==0)
            throw new IllegalArgumentException();
        for (int i=0; i<layers.length; i++){
            if (layers[i]==null || (i>1 && layers[i].getInputSize()!=layers[i-1].getSize()))
                throw new IllegalArgumentException();
        }

        this.layers=layers;
    }

    public Network(){}

    public final Size getInputSize(){
        return layers[0].getInputSize();
    }

    public final Size getOutputSize(){
        return layers[layers.length-1].getSize();
    }

    public final int getSize(){
        return layers.length;
    }

    public Layer getLayer(int index){
        return layers[index];
    }

    /*public double[][][] computeOutput(double[][][] input){
        if (input==null || !getInputSize().compare(new Size(input)))
            throw new IllegalArgumentException();

        double[][][] output = input;
        final int size = layers.length;
        for (int i=0; i<size; i++){
            output=layers[i].computeOutput(output);
        }
        return output;
    }*/

    public void saveToFile(String fileName){
        if (fileName==null) throw new IllegalArgumentException();
        try{
            ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(fileName));
            outputStream.writeObject(this);
            outputStream.close();
        }catch (Exception e) {
            throw new IllegalArgumentException(e);
        }
    }

    public static Network loadFromFile(String fileName){
        if (fileName==null) throw new IllegalArgumentException();
        Object network=null;
        try{
            ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(fileName));
            network = inputStream.readObject();
            inputStream.close();
        }
        catch (Exception e) {
            throw new IllegalArgumentException(e);
        }
        return (Network)network;
    }
}
