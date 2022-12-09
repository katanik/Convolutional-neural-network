package com.company.neural_network;

import java.io.Serializable;

public class Size implements Serializable{
    public int width, height, depth;

    public Size(int depth, int width, int height) {
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    public Size(double[][][] a) {
        if (a == null) {
            width = 0;
            height = 0;
            depth = 0;
        } else {
            depth = a.length;
            height = a[0].length;
            width = a[0][0].length;
        }
    }

    public boolean compare(Size size){
        return width==size.width && height==size.height && depth==size.depth;
    }

    public boolean isPositive(){
        return width>0 && height>0 && depth>0;
    }

    public void set(int addDepth, int addWidth, int addHeight){
        depth=addDepth;
        width=addWidth;
        height=addHeight;
    }
}
