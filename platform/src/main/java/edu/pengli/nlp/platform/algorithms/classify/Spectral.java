package edu.pengli.nlp.platform.algorithms.classify;

import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;

public class Spectral extends Clusterer {

	Metric metric;
	int numClusters;
	MatlabProxy proxy;

	public Spectral(Pipe instancePipe, int numClusters, Metric metric,
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
				similarityMatrix[i][j] = metric.distance(fv_i, fv_j);
			}
		}

		int clusterLabels[] = new int[instances.size()];
		
		try {
			processor.setNumericArray("A", new MatlabNumericArray(
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

			proxy.eval("L=D-A");
			proxy.eval("L=D^(-0.5)*L*D^(-0.5)");
			proxy.eval("[U,DV]=eig(L)");
			proxy.eval("[eigval, idx] = sort(diag(DV))");
			proxy.eval("U= U(:, idx(1:" + numClusters + "))");
			proxy.eval("U=U./repmat(sqrt(sum(U.*U,2)),1," + numClusters + ")");
			proxy.eval("labels = kmeans(U," + numClusters + ",'Replicates',20)");
			
/*			FeatureVector vec = (FeatureVector)instances.get(0).getData();
			int dimension = vec.getValues().length;
			double[][] dataMatrix = new double[instances.size()][dimension];
			for (int i = 0; i < instances.size(); i++) {
				FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
				for(int j=0; j<dimension; j++)
					dataMatrix[i][j] = fv_i.getValues()[j];
			}
			
			processor.setNumericArray("arr", new MatlabNumericArray(
					dataMatrix, null));
			proxy.eval("labels = kmeans(arr,"+numClusters+")");*/

			double[][] labels = processor.getNumericArray("labels")
					.getRealArray2D();

			for (int i = 0; i < instances.size(); i++)
				clusterLabels[i] = (int) labels[i][0]-1;		
			
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return new Clustering(instances, numClusters, clusterLabels);

	}

}
