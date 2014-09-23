package edu.pengli.nlp.platform.algorithms.ranking;

import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.Maths;
import edu.pengli.nlp.platform.util.matrix.Matrix;

public class LexRank {

	private InstanceList instances;
	private Matrix matrix;
	private double threshold;
	private double damping;
	private double error;
	private int[] degree;
	private double[] L;
	private double[][] cosineMatrix;

	public LexRank(InstanceList instances, double threshold, double damping,
			double error) {
		this.instances = instances;
		this.threshold = threshold;
		this.damping = damping;
		this.error = error;

	}

	public void rank() {
		int N = instances.size();
		degree = new int[N];
		L = new double[N];
		cosineMatrix = new double[N][N];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cosineMatrix[i][j] = Maths.idf_modified_cosine(instances.get(i),
						instances.get(j));
				if (cosineMatrix[i][j] > threshold) {
					cosineMatrix[i][j] = 1;
					degree[i]++;
				} else
					cosineMatrix[i][j] = 0;
			}
		}

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if (degree[i] == 0) {
					cosineMatrix[i][j] = damping / N; // prevent NAN
				} else {
					cosineMatrix[i][j] = damping / N + (1 - damping)
							* cosineMatrix[i][j] / degree[i]; // prevent NAN
				}

			}
		}

		matrix = new Matrix(cosineMatrix);

		L = PowerMethod(error, N);

	}

	private double[] PowerMethod(double error, int N) {

		Matrix p0 = new Matrix(N, 1);
		Matrix p1 = new Matrix(N, 1);

		for (int i = 0; i < N; i++) {
			p0.set(i, 0, 1 / (double) N);
		}

		Matrix mT = matrix.transpose();
		Matrix pMinus;
		p1 = mT.times(p0);
		pMinus = p1.minus(p0);

		while (pMinus.getMax() >= error) {
			p0 = p1;
			p1 = mT.times(p0);
			pMinus = p1.minus(p0);
		}

		return p1.getCol(0);
	}

	public double[] getScore() {
		return L;
	}

}