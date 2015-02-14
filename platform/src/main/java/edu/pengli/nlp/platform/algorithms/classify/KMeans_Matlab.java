package edu.pengli.nlp.platform.algorithms.classify;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;

public class KMeans_Matlab extends Clusterer{
	
	Metric metric;
	int numClusters;
	MatlabProxy proxy;
	
	public KMeans_Matlab(Pipe instancePipe, int numClusters, Metric metric,
			MatlabProxy proxy) {

		super(instancePipe);

		this.metric = metric;
		this.numClusters = numClusters;
		this.proxy = proxy;

	}

	public Clustering cluster(InstanceList instances) {
				
		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
		
		FeatureVector vec = (FeatureVector)instances.get(0).getData();
		int dimension = vec.getValues().length;
		
		double[][] dataMatrix = new double[instances.size()][dimension];
		for (int i = 0; i < instances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
			for(int j=0; j<dimension; j++)
				dataMatrix[i][j] = fv_i.getValues()[j];
		}
		
		int clusterLabels[] = new int[instances.size()];
		try {
			
			processor.setNumericArray("arr", new MatlabNumericArray(
					dataMatrix, null));
			proxy.eval("labels = kmeans(arr,"+numClusters+ ",'Replicates',20)");
			double[][] labels = processor.getNumericArray("labels").getRealArray2D();
			for (int i = 0; i < instances.size(); i++)
				clusterLabels[i] = (int) labels[i][0]-1;
			
			
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return new Clustering(instances, numClusters, clusterLabels);
	}
	
}
