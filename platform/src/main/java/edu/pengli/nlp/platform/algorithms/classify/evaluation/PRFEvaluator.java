package edu.pengli.nlp.platform.algorithms.classify.evaluation;

import java.util.HashSet;

import edu.pengli.nlp.platform.algorithms.classify.Clustering;

public class PRFEvaluator implements ClusteringEvaluator {
	
	int precisionNumerator;
	int precisionDenominator;
	int recallNumerator;
	int recallDenominator;
	
	public PRFEvaluator() {
		precisionNumerator = precisionDenominator = recallNumerator = recallDenominator = 0;
	}

	public String evaluate(Clustering truth, Clustering predicted) {
		double[] vals = getEvaluationScores(truth, predicted);
		return "pr=" + vals[0] + " re=" + vals[1] + " f1=" + vals[2];
	}


	public double[] getEvaluationScores(Clustering truth, Clustering predicted) {
		// Precision = \sum_i [ |siprime| - |pOfsiprime| ] / \sum_i [ |siprime| - 1 ]		
		// where siprime is a predicted cluster, pOfsiprime is the set of
		// true clusters that contain elements of siprime.
		int numerator = 0;
		int denominator = 0;
		for (int i = 0; i < predicted.getNumClusters(); i++) {
			int[] siprime = predicted.getIndicesWithLabel(i);
			HashSet<Integer> pOfsiprime = new HashSet<Integer>();
			for (int j = 0; j < siprime.length; j++) 
				pOfsiprime.add(truth.getLabel(siprime[j]));
			numerator += siprime.length - pOfsiprime.size();
			denominator += siprime.length - 1;
		}
		precisionNumerator += numerator;
		precisionDenominator += denominator;
		double precision = (double)numerator / denominator;

		// Recall = \sum_i [ |si| - |pOfsi| ] / \sum_i [ |si| - 1 ]		
		// where si is a true cluster, pOfsi is the set of predicted
		// clusters that contain elements of si.
		numerator = denominator = 0;
		for (int i = 0; i < truth.getNumClusters(); i++) {
			int[] si = truth.getIndicesWithLabel(i);
			HashSet<Integer> pOfsi = new HashSet<Integer>();
			for (int j = 0; j < si.length; j++) 
				pOfsi.add(new Integer(predicted.getLabel(si[j])));
			numerator += si.length - pOfsi.size();
			denominator += si.length - 1;
		}
		recallNumerator += numerator;
		recallDenominator += denominator;
		double recall = (double)numerator / denominator;
		return new double[]{precision,recall,(2 * precision * recall / (precision + recall))};
	}

}
