package edu.pengli.nlp.platform.algorithms.classify;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;

public class LabelPropagationSemiSupervisedClustering 
				extends SemiSupervisedClustering{
	
	public LabelPropagationSemiSupervisedClustering(Pipe instancePipe,
			InstanceList seeds, Metric metric, MatlabProxy proxy) {
		super(instancePipe, seeds, metric, proxy);

	}

	public Clustering cluster(InstanceList instances) {

		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);

		double[][] weightMatrix = new double[instances.size()+seeds.size()][instances
				.size()+seeds.size()];
		InstanceList allInsts = new InstanceList(null);
		for(Instance seed : seeds)
			allInsts.add(seed);
		for(Instance inst : instances)
			allInsts.add(inst);
		

		for (int i = 0; i < allInsts.size(); i++) {
			FeatureVector fv_i = (FeatureVector) allInsts.get(i).getData();
			for (int j = 0; j < allInsts.size(); j++) {
				FeatureVector fv_j = (FeatureVector) allInsts.get(j).getData();
/*				double sum = 0.0;
				for(int k=0; k<fv_i.getValues().length; k++){
					sum += Math.pow((fv_i.getValues()[k]-fv_j.getValues()[k]), 2)/10;
				}
				weightMatrix[i][j] = Math.exp(-sum);*/
				weightMatrix[i][j] = 1-metric.distance(fv_i, fv_j);
//				weightMatrix[i][j] = metric.distance(fv_i, fv_j);
			}
		}
		int clusterLabels[] = new int[instances.size()];
		try {
			
			
			processor.setNumericArray("A", new MatlabNumericArray(
					weightMatrix, null));

			
			double[][] Y = new double[instances.size()+seeds.size()][seeds.size()];
			for (int i = 0; i < instances.size()+seeds.size(); i++) {
				for (int j = 0; j < seeds.size(); j++) {
						if(i == j)
							Y[i][j] = 1.0;
						else
							Y[i][j] = 0.0;
				}
			}

			processor.setNumericArray("Y", new MatlabNumericArray(
					Y, null));
			proxy.eval("obj = zeros(20,1)");
			proxy.eval("D = diag(sum(A,2))");		
			proxy.eval("n = size(A,1)");
			proxy.eval("u = zeros(n,1)");
			proxy.eval("y = sum(Y,2)");
			proxy.eval("u(y==1) = 10000000");
			proxy.eval("U = diag(u)");
			proxy.eval("L = D - A");
			proxy.eval("F = mldivide((L+U),(U*Y))");
			proxy.eval("X = F'");
			if(processor.getNumericArray("X").getLength() == 1){
				proxy.eval("F' = [F'; zeros(1,size(F',2))]");
				proxy.eval("F' = [F'; zeros(1,size(F',2))]");
			}
			proxy.eval("aa=sum(F'.*F')");
			proxy.eval("bb=sum(F'.*F')");
			proxy.eval("ab=F*F'");
			proxy.eval("d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab");
			proxy.eval("d = real(d)");
			proxy.eval("W = sqrt(abs(d)+eps)");
			for(int i=1; i<=20; i++){
				proxy.eval("A1 = A./(2*W)");
				proxy.eval("A1 = (A1+A1')/2");
				proxy.eval("d1 = sum(A1,2)");
				proxy.eval("D1 = diag(d1)");
				proxy.eval("L1 = D1 - A1");
				proxy.eval("F = mldivide((L1+U),(U*Y))");
				proxy.eval("X = F'");
				if(processor.getNumericArray("X").getLength() == 1){
					proxy.eval("F' = [F'; zeros(1,size(F',2))]");
					proxy.eval("F' = [F'; zeros(1,size(F',2))]");
				}
				proxy.eval("aa=sum(F'.*F')");
				proxy.eval("bb=sum(F'.*F')");
				proxy.eval("ab=F*F'");
				proxy.eval("d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab");
				proxy.eval("d = real(d)");
				proxy.eval("W = sqrt(abs(d)+eps)");
				proxy.eval("obj("+i+") = 0.5*sum(sum((A.*W)))+trace((F-Y)'*U*(F-Y))");
			}
			
			double[][] F = processor.getNumericArray("F").getRealArray2D();
			for(int i=0; i<instances.size(); i++){
				double max = Double.MIN_VALUE;
				int idx = -1;
				for(int j=0; j<seeds.size(); j++){
					if(max < F[i+seeds.size()][j]){
						max = F[i+seeds.size()][j];
						idx = j;
					}
				}
				clusterLabels[i] = idx;	
			}
							
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return new Clustering(instances, seeds.size(), clusterLabels);

	}
}
