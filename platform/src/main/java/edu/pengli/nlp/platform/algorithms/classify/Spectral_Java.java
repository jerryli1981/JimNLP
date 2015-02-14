package edu.pengli.nlp.platform.algorithms.classify;


import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.jblas.Eigen;

import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;


public class Spectral_Java extends Clusterer {

	Metric metric;
	int numClusters;

	public Spectral_Java(Pipe instancePipe, int numClusters, Metric metric) {

		super(instancePipe);

		this.metric = metric;
		this.numClusters = numClusters;

	}

	public Clustering cluster(InstanceList instances) {

		DoubleMatrix w = new DoubleMatrix(instances.size(), instances.size());
		for (int i = 0; i < instances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
			for (int j = 0; j < instances.size(); j++) {
				FeatureVector fv_j = (FeatureVector) instances.get(j).getData();
				w.put(i, j, 1-metric.distance(fv_i, fv_j));
			}
		}

		System.err.println("start");
		System.err.println("Matrix construction...");
		convertToNormalizedLaplacian(w);
		System.err.println("Eigen value decomposition...");
		final DoubleMatrix[] res = getEigenValues(w);
		final double[][] pts = new double[w.getColumns()][numClusters];
		for (int i = 1; i <= numClusters; i++) {
			for (int j = 0; j < w.getColumns(); j++)
				pts[j][i - 1] = res[0].get(j, i);
		}
		
		InstanceList tmpList = new InstanceList(null);
		for(int i=0; i<w.getColumns(); i++){
			double[] val = new double[numClusters];
			int[] idx = new int[numClusters];
			for(int j=0; j<numClusters; j++){
				val[j] = pts[i][j];
				idx[j] = j;
			}
			FeatureVector fv = new FeatureVector(idx, val);
			Instance inst = new Instance(fv, null, null, instances.get(i).getSource());
			tmpList.add(inst);
		}
		
		KMeans_Java kmeans = new KMeans_Java(new Noop(), numClusters, metric);
		Clustering predicted = kmeans.cluster(tmpList);
		
		return predicted;

/*		final int[][] result = kmeans.getClusters(pts, numClusters);

		int clusterLabels[] = new int[instances.size()];
		for (int i = 0; i < instances.size(); i++){
			for(int j=0; j<numClusters; j++){
				int[] ids = result[j];
				ArrayList<Integer> iList = new ArrayList<Integer>();
				for(Integer k : ids){
					iList.add(k);
				}
				if(iList.contains(i)){
					clusterLabels[i] = j;
					break;
				}
			}
		}
				

		return new Clustering(instances, numClusters, clusterLabels);*/

	}

	private DoubleMatrix[] getEigenValues(DoubleMatrix w) {

		final DoubleMatrix[] result = Eigen.symmetricEigenvectors(w);
		return result;
	}

	private void convertToNormalizedLaplacian(DoubleMatrix w) {
		final int n = w.getColumns();
		double[] d2 = getD(w, n);
		for (int i = 0; i < n; i++)
			if (d2[i] != 0)
				d2[i] = 1 / Math.sqrt(d2[i]);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				w.put(i, j, 1. - d2[i] * w.get(i, j) * d2[j]);
			}
	}

	private double[] getD(DoubleMatrix w, int n) {
		double[] d2 = new double[n];
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				d2[i] += w.get(i, j);
		return d2;
	}

	private void convertToUnnormalizedLaplacian(DoubleMatrix w) {
		final int n = w.getColumns();
		double[] d = getD(w, n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				w.put(i, j, (i == j ? d[i] : 0) - w.get(i, j));
			}
	}

	private class Kmeans {
		private double centroinds[][];
		private List<Integer> clusters[];
		private int inside[];

		private void recalcCentroinds(double vec[][], int k) {
			for (int i = 0; i < k; ++i) {
				double sum[] = new double[vec[0].length];
				for (int j = 0; j < sum.length; ++j) {
					sum[j] = 0;
				}
				double n = clusters[i].size();
				for (int t : clusters[i])
					for (int j = 0; j < vec[t].length; ++j)
						sum[j] += vec[t][j];
				for (int j = 0; j < sum.length; ++j)
					sum[j] /= n;
				centroinds[i] = sum;
			}
			for (int i = 0; i < clusters.length; ++i)
				clusters[i].clear();
		}

		private int getNearest(double a[]) {
			double min = sub(a, centroinds[0]);
			int res = 0;
			for (int i = 1; i < centroinds.length; ++i) {
				double t = sub(a, centroinds[i]);
				if (t < min) {
					min = t;
					res = i;
				}
			}
			return res;
		}

		private double sub(double a[], double b[]) {
			if (a.length != b.length)
				return -1;
			double res = 0;
			for (int i = 0; i < a.length; ++i)
				res += (a[i] - b[i]) * (a[i] - b[i]);
			return res;
		}

		private boolean rebuildclusters(double vec[][], int k) {
			boolean flag = true;
			for (int i = 0; i < vec.length; ++i) {
				int t = getNearest(vec[i]);
				clusters[t].add(i);
				if (inside[i] != t) {
					flag = false;
					inside[i] = t;
				}
			}
			if (flag)
				return true;
			return false;
		}

		public int[][] getClusters(double vec[][], int k) {
			centroinds = new double[k][vec[0].length];
			inside = new int[vec.length];
			for (int i = 0; i < vec.length; ++i) {
				inside[i] = 0;
			}
			clusters = new List[k];
			for (int i = 0; i < k; ++i) {
				clusters[i] = new ArrayList<Integer>();
			}
			if (vec.length < k) {
				return null;
			}
			for (int i = 0; i < k; ++i) {
				centroinds[i] = vec[i];
			}
			int numIterations = 0;
			while (!rebuildclusters(vec, k) && numIterations < 1000) {
				recalcCentroinds(vec, k);
				numIterations++;
				System.err.println(numIterations);
			}
			System.err.println("numIterations: " + numIterations);
			final int[][] result = new int[k][];
			for (int i = 0; i < result.length; i++) {
				final List<Integer> currentCluster = clusters[i];
				result[i] = new int[currentCluster.size()];
				for (int j = 0; j < currentCluster.size(); j++)
					result[i][j] = currentCluster.get(j);
			}
			return result;
		}
	}

}
