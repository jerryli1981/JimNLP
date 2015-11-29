package edu.pengli.nlp.platform.algorithms.classify;

import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;
import edu.pengli.nlp.platform.util.Maths;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;

public class Spectral_Matlab_Cos extends Clusterer {

	Metric metric;
	int numClusters;
	MatlabProxy proxy;

	public Spectral_Matlab_Cos(Pipe instancePipe, int numClusters, Metric metric,
			MatlabProxy proxy) {

		super(instancePipe);

		this.metric = metric;
		this.numClusters = numClusters;
		this.proxy = proxy;

	}

	public Clustering cluster(InstanceList instances) {

		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);

		double[][] similarityMatrix = new double[instances.size()][instances
				.size()];
		
		for (int i = 0; i < instances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
			for (int j = 0; j < instances.size(); j++) {
				FeatureVector fv_j = (FeatureVector) instances.get(j).getData();
/*				double sum = 0.0;
				for(int k=0; k<fv_i.getValues().length; k++){
					sum += Math.pow((fv_i.getValues()[k]-fv_j.getValues()[k]), 2)/10;
				}
				similarityMatrix[i][j] = Math.exp(-sum); */
				similarityMatrix[i][j] = 1-metric.distance(fv_i, fv_j);
			}
		}
		
		// 25, ROUGE-1, 0.30184
		// 25, ROUGE-1, 0.30707 below
/*		int N = instances.size();
		int[] degree = new int[N];
		double threshold = 0.5;
		double damping = 0.5;
		for (int i = 0; i < instances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
			for (int j = 0; j < instances.size(); j++) {
				FeatureVector fv_j = (FeatureVector) instances.get(j).getData();
				similarityMatrix[i][j] = 1-metric.distance(fv_i, fv_j);
				if (similarityMatrix[i][j] > threshold) {
					similarityMatrix[i][j] = 1;
					degree[i]++;
				} else
					similarityMatrix[i][j] = 0;
			}
		}

		for (int i = 0; i < instances.size(); i++) {
			for (int j = 0; j < instances.size(); j++) {
				if (degree[i] == 0) {
					similarityMatrix[i][j] = damping / N; // prevent NAN
				} else {
					similarityMatrix[i][j] = damping / N + (1 - damping)
							* similarityMatrix[i][j] / degree[i]; // prevent NAN
				}

			}
		}*/
		
		int clusterLabels[] = new int[instances.size()];
		Clustering predicted = null;
		
		try {
			processor.setNumericArray("W", new MatlabNumericArray(
					similarityMatrix, null));

			double[][] diagonalMatrix = new double[instances.size()][instances
					.size()];
			for (int i = 0; i < instances.size(); i++) {
				double sum = 0.0;
				for (int j = 0; j < instances.size(); j++) {
					sum += similarityMatrix[i][j];
				}
				diagonalMatrix[i][i] = sum;
			}
			processor.setNumericArray("D", new MatlabNumericArray(
					diagonalMatrix, null));

			proxy.eval("L=D-W");
			proxy.eval("L=D^(-0.5)*L*D^(-0.5)");
			proxy.eval("[U,DV]=eig(L)");
			proxy.eval("[eigval, idx] = sort(diag(DV))");
			proxy.eval("U= U(:, idx(1:" + numClusters + "))");
			proxy.eval("U=U./repmat(sqrt(sum(U.*U,2)),1," + numClusters + ")");
			proxy.eval("labels = kmeans(U," + numClusters + ",'Replicates',20)");
			
/*			double[][] U = processor.getNumericArray("U")
					.getRealArray2D();
			
			InstanceList tmpList = new InstanceList(null);
			for(int i=0; i<instances.size(); i++){
				double[] val = new double[numClusters];
				int[] idx = new int[numClusters];
				for(int j=0; j<numClusters; j++){
					val[j] = U[i][j];
					idx[j] = j;
				}
				FeatureVector fv = new FeatureVector(idx, val);
				Instance inst = new Instance(fv, null, null, instances.get(i).getSource());
				tmpList.add(inst);
			}
			
			KMeans kmeans = new KMeans(new Noop(), numClusters, metric);
			predicted = kmeans.cluster(tmpList);*/
			

			double[][] labels = processor.getNumericArray("labels")
					.getRealArray2D();

			for (int i = 0; i < instances.size(); i++)
				clusterLabels[i] = (int) labels[i][0]-1;		
			
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return new Clustering(instances, numClusters, clusterLabels);
//		return predicted;
	}

}
