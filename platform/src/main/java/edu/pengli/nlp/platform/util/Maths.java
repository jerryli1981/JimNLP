/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

/** 
 @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.pengli.nlp.platform.util;

import java.util.ArrayList;

import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;

public final class Maths {

	/**
	 * Returns the KL divergence, K(p1 || p2).
	 * 
	 * The log is w.r.t. base 2.
	 * <p>
	 * 
	 * *Note*: If any value in <tt>p2</tt> is <tt>0.0</tt> then the
	 * KL-divergence is <tt>infinite</tt>.
	 * 
	 */
	public static double klDivergence(double[] p1, double[] p2) {
		assert (p1.length == p2.length);
		double klDiv = 0.0;
		for (int i = 0; i < p1.length; ++i) {
			if (p1[i] == 0) {
				continue;
			}
			if (p2[i] == 0) {
				return Double.POSITIVE_INFINITY;
			}
			klDiv += p1[i] * Math.log(p1[i] / p2[i]);
		}
		return klDiv / Math.log(2); 
	}

	/**
	 * Returns the Jensen-Shannon divergence.
	 */
	public static double jensenShannonDivergence(double[] p1, double[] p2) {
		assert (p1.length == p2.length);
		double[] average = new double[p1.length];
		for (int i = 0; i < p1.length; ++i) {
			average[i] += (p1[i] + p2[i]) / 2;
		}
		return (klDivergence(p1, average) + klDivergence(p2, average)) / 2;
	}

	/** sqrt(a^2 + b^2) without under/overflow. **/

	public static double hypot(double a, double b) {
		double r;
		if (Math.abs(a) > Math.abs(b)) {
			r = b / a;
			r = Math.abs(a) * Math.sqrt(1 + r * r);
		} else if (b != 0) {
			r = a / b;
			r = Math.abs(b) * Math.sqrt(1 + r * r);
		} else {
			r = 0.0;
		}
		return r;
	}
	
	public static double idf_modified_cosine(Instance inst_i, Instance inst_j) {

		double sim = 0.0;

		FeatureVector[] tf_idf_fv_i = (FeatureVector[]) inst_i.getData();
		FeatureVector tf_fv_i = tf_idf_fv_i[0];
		FeatureVector idf_fv_i = tf_idf_fv_i[1];
		int[] index_i = tf_fv_i.getIndices();
		double sum_i = 0;
		for (int i = 0; i < index_i.length; i++) {
			int idx = index_i[i];
			double tf_i = tf_fv_i.getValues()[tf_fv_i.location(idx)];
			double idf_i = idf_fv_i.getValues()[idf_fv_i.location(idx)];
			sum_i += Math.pow(tf_i * idf_i, 2);
		}
		double sqrt_sum_i = Math.sqrt(sum_i);

		FeatureVector[] tf_idf_fv_j = (FeatureVector[]) inst_j.getData();
		FeatureVector tf_fv_j = tf_idf_fv_j[0];
		FeatureVector idf_fv_j = tf_idf_fv_j[1];
		int[] index_j = tf_fv_j.getIndices();
		double sum_j = 0;
		for (int i = 0; i < index_j.length; i++) {
			int idx = index_j[i];
			double tf_j = tf_fv_j.getValues()[tf_fv_j.location(idx)];
			double idf_j = idf_fv_j.getValues()[idf_fv_j.location(idx)];
			sum_j += Math.pow(tf_j * idf_j, 2);
		}
		double sqrt_sum_j = Math.sqrt(sum_j);

		ArrayList<Integer> commonIdx = new ArrayList<Integer>();

		for (int i = 0; i < index_i.length; i++) {
			int x = index_i[i];
			for (int j = 0; j < index_j.length; j++) {
				int y = index_j[j];
				if (x == y)
					commonIdx.add(x);
			}
		}

		double sum = 0;
		for (int i = 0; i < commonIdx.size(); i++) {
			int idx = commonIdx.get(i);
			double tf_i = tf_fv_i.getValues()[tf_fv_i.location(idx)];
			double tf_j = tf_fv_j.getValues()[tf_fv_j.location(idx)];
			double idf = idf_fv_i.getValues()[idf_fv_i.location(idx)];
			sum += tf_i * tf_j * Math.pow(idf, 2);
		}

		sim = (sum+1) / (sqrt_sum_i * sqrt_sum_j);
		return sim;
	}

}
