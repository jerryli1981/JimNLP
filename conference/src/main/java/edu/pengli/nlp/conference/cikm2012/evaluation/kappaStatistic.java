package edu.pengli.nlp.conference.cikm2012.evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class kappaStatistic {
	static String[] labels = { "0", "1", "2", "3" };

	static double kappa(double YY, double YN, double NY, double NN, double Total) {
		double value = 0.0d;

		double Pa = (YY + NN) / Total;
		double Pnr = (NY + NN + YN + NN) / (Total + Total);
		double Pr = (YY + YN + YY + NY) / (Total + Total);
		double Pe = Math.pow(Pnr, 2) + Math.pow(Pr, 2);
		value = (Pa - Pe) / (1 - Pe);
		return value;
	}

	public static void main(String[] args) throws IOException {
		double YY = 0.0d;
		double YN = 0.0d;
		double NY = 0.0d;
		double NN = 0.0d;
		double Total = 0;
		
		for (int i = 0; i < labels.length; i++) {
			BufferedReader LP = new BufferedReader(
					new FileReader(
							"../data/EMNLP2012/Output/summary/LabeledByMe"));
			BufferedReader JJ = new BufferedReader(
					new FileReader(
							"../data/EMNLP2012/Output/summary/LabeledByGao"));
			String slp = null;
			String sjj = null;
			String label = labels[i];

			while ((slp = LP.readLine()) != null
					&& (sjj = JJ.readLine()) != null) {
				if (slp.startsWith("<SP score=")) { // <SP score= 1 >
					Total++;
                    slp = slp.replace("<SP score=", "");
                    sjj = sjj.replace("<SP score=", "");
					String lp = slp.replace(">", "").trim();
					String jj = sjj.replace(">", "").trim();
					if (jj.contains(label) && lp.contains(label)) {
						YY++;
					} else if (jj.contains(label) && !lp.contains(label)) {
						YN++;
					} else if (!jj.contains(label) && lp.contains(label)) {
						NY++;
					} else if (!jj.contains(label) && !lp.contains(label)) {
						NN++;
					}

				}
			}
		}
		
		double value = kappa(YY, YN, NY, NN, Total);
		System.out.println( value);

	}

}
