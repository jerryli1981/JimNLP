package edu.pengli.nlp.platform.algorithms.classify;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;

public class Spectral_CappedNorm extends Clusterer {
	
	Metric metric;
	int numClusters;
	MatlabProxy proxy;

	public Spectral_CappedNorm(Pipe instancePipe, int numClusters, Metric metric,
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
				double sum = 0.0;
				for(int k=0; k<fv_i.getValues().length; k++){
					sum += Math.pow((fv_i.getValues()[k]-fv_j.getValues()[k]), 2)/10;
				}
				similarityMatrix[i][j] = Math.exp(-sum); 
//				similarityMatrix[i][j] = 1-metric.distance(fv_i, fv_j);
			}
		}
		
		double[][] D = new double[instances.size()][instances.size()];
		for (int i = 0; i < instances.size(); i++) {
			for (int j = 0; j < instances.size(); j++) {
					if(i == j)
						D[i][j] = 1.0;
					else
						D[i][j] = 0.0;
			}
		}
		

		
		int clusterLabels[] = new int[instances.size()];
	
		try {
			processor.setNumericArray("A", new MatlabNumericArray(
					similarityMatrix, null));
			
			processor.setNumericArray("D", new MatlabNumericArray(
					D, null));

			proxy.eval("obj = zeros(20,1)");
			proxy.eval("[v d] = eig(A, D)");
			proxy.eval("[d idx] = sort(d,'descend')");
			proxy.eval("F = v(:,idx(1:"+numClusters+"))");
			proxy.eval("F = F*diag(sqrt(1./diag(F'*D*F)))");
			proxy.eval("X = F'");
			if(processor.getNumericArray("X").getLength() == 1){
				proxy.eval("F' = [F'; zeros(1,size(F',2))]");
				proxy.eval("F' = [F'; zeros(1,size(F',2))]");
			}
			proxy.eval("aa=sum(F'.*F')");
			proxy.eval("bb=sum(F'.*F')");
			proxy.eval("ab=F*F'");
			proxy.eval("d1 = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab");
			proxy.eval("d1 = real(d1)");
			proxy.eval("W = sqrt(abs(d1)+eps)");
			for(int iter = 1; iter <= 20; iter++){
				proxy.eval("A1 = A./W");
				proxy.eval("A1 = (A1+A1')/2");
				proxy.eval("L1 = diag(sum(A1)) - A1");
				proxy.eval("[v d] = eig(L1, D)");
				proxy.eval("d = diag(d)");
				proxy.eval("[d idx] = sort(d)");
				proxy.eval("F = v(:,idx(1:"+numClusters+"))");
				proxy.eval("F = F*diag(sqrt(1./diag(F'*D*F)))");
				proxy.eval("X = F'");
				if(processor.getNumericArray("X").getLength() == 1){
					proxy.eval("F' = [F'; zeros(1,size(F',2))]");
					proxy.eval("F' = [F'; zeros(1,size(F',2))]");
				}
				proxy.eval("aa=sum(F'.*F')");
				proxy.eval("bb=sum(F'.*F')");
				proxy.eval("ab=F*F'");
				proxy.eval("d2 = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab");
				proxy.eval("d2 = real(d2)");
				proxy.eval("W = sqrt(abs(d2)+eps)");
//				proxy.eval("obj("+iter+") = sum(sum((A.*W)))");	
			}
			
			double[][] F = processor.getNumericArray("F").getRealArray2D();
			for(int i=0; i<instances.size(); i++){
				double max = Double.NEGATIVE_INFINITY;
				int idx = -1;
				for(int j=0; j<numClusters; j++){
					if(max < F[i][j]){
						max = F[i][j];
						idx = j;
					}
				}
				clusterLabels[i] = idx;	
			}

				
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return new Clustering(instances, numClusters, clusterLabels);
	}

}

