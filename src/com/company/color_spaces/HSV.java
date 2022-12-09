package com.company.color_spaces;


public class HSV {
    private double H, S, V;

    public HSV(){}
    public HSV(RGB rgb){setHSV(rgb);}

    public void setHSV(RGB rgb) {
        double R = rgb.getR(), G = rgb.getG(), B = rgb.getB();
        double max = Math.max(Math.max(R, G), B);
        double min = Math.min(Math.min(R, G), B);
        if (max != min) {
            if (max == R) {
                if (G >= B)
                    H = 60. * ((G - B) / (max - min));
                if (G < B)
                    H = 60. * ((G - B) / (max - min)) + 360.;
            } else {
                if (max == G)
                    H = 60. * ((B - R) / (max - min)) + 120.;
                else
                    H = 60. * ((R - G) / (max - min)) + 240.;
            }
        } else
            H = 0.;
        S = (max == 0. ? 0. : 1. - min / max)*100.;
        V = max*100./255.;
    }

    public void changeV(double add){
        if (add<0)
            V*=(1+add/100.);
        else
            V+=(100.-V)*add/100.;
    }

    public void changeS(double add){
        if (add<0)
            S*=(1+add/100.);
        else
            S+=(100.-S)*add/100.;
    }

    public void changeH(double add){
        H+=add;
        if (H>360) H-=360;
        if (H<0) H+=360;
    }

    public String out() {
        return "hue = " + Double.toString(H) + ",  saturation = " + Double.toString(S) + ",  value = " + Double.toString(V);
    }

    public String outInteger() {
        return "hue = " + Integer.toString((int)Math.round(H)) +
                ",  saturation = " + Integer.toString((int)Math.round(S)) + "%" +
                ",  value = " + Integer.toString((int)Math.round(V)) + "%";
    }

    public RGB toRGB(){
        H/=360.;
        S/=100.;
        V/=100.;
        if (S==0)
            return new RGB(V*255., V*255., V*255.);

        double var_h=H*6.;
        if (var_h==6) var_h=0;
        double var_i=Math.floor(var_h);
        double var_1=V*(1-S);
        double var_2= V*(1-S*(var_h-var_i));
        double var_3 = V*(1-S*(1-(var_h-var_i)));


        switch ((int)var_i){
            case 0 : return new RGB(255.*V, 255.*var_3, 255.*var_1);
            case 1 : return new RGB(255.*var_2, 255.*V, 255.*var_1);
            case 2 : return new RGB(255.*var_1, 255.*V, 255.*var_3);
            case 3 : return new RGB(255.*var_1, 255.*var_2, 255.*V);
            case 4 : return new RGB(255.*var_3, 255.*var_1, 255.*V);
            default: return new RGB(255.*V, 255.*var_1, 255.*var_2);
        }


       /* double Vmin=(100.-S)*V/100.;
        double a=(V-Vmin)*((int)H%60)/60.;
        double Vinc=Vmin+a;
        double Vdec=V-a;

        V*=255./100.;
        Vinc*=255./100.;
        Vdec*=255./100.;

        switch (((int)H/60)%6){
            case 0 : return new RGB(V, Vinc, Vmin);
            case 1 : return new RGB(Vdec, V, Vmin);
            case 2 : return new RGB(Vmin, V, Vinc);
            case 3 : return new RGB(Vmin, Vdec, V);
            case 4 : return new RGB(Vinc, Vmin, V);
            case 5 : return new RGB(V, Vmin, Vdec);
        }
        return new RGB(-1, -1, -1);
*/

    }
}
