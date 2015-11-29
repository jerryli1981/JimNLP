package edu.pengli.nlp.platform.algorithms.classify.evaluation;

import edu.pengli.nlp.platform.algorithms.classify.Clustering;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;


public class PurityEvaluator implements ClusteringEvaluator {

	private double getNumOverlap(InstanceList predictCluster, InstanceList trueCluster) {
		double numOverlap = 0;
		for (int i = 0; i < predictCluster.size(); i++) {
			Instance predictInstance = predictCluster.get(i);
			for (int j = 0; j < trueCluster.size(); j++) {
				Instance trueInstance = trueCluster.get(j);
				if (predictInstance.equals(trueInstance)) {
					numOverlap++;
				}

			}
		}

		return numOverlap;
	}

	public String evaluate(Clustering truth, Clustering predicted) {

		double totalNumOverlap = 0;
		double TotalNumofInstance = truth.getNumInstances();
		for (int i = 0; i < predicted.numLabels; i++) {
			InstanceList predictCluster = predicted.getCluster(i);
			double maxNumOverlap = 0;
			for (int j = 0; j < truth.numLabels; j++) {
				InstanceList trueCluster = truth.getCluster(j);
				double numOverlap = getNumOverlap(predictCluster, trueCluster);
				if (maxNumOverlap <= numOverlap) {
					maxNumOverlap = numOverlap;
				}
			}
			totalNumOverlap += maxNumOverlap;
		}
		double purity = totalNumOverlap / TotalNumofInstance;
		return Double.toString(purity);
	}


	public double[] getEvaluationScores(Clustering truth, Clustering predicted) {
		// TODO Auto-generated method stub
		return null;
	}

}
