package edu.pengli.nlp.platform.algorithms.classify;

import java.io.IOException;
import java.util.ArrayList;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLDouble;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;

public class LabelPropagationSemiSupervisedClustering 
				extends SemiSupervisedClustering{
	int sigma = 0;
	public LabelPropagationSemiSupervisedClustering(Pipe instancePipe,
			InstanceList seeds, Metric metric, MatlabProxy proxy, int sigma) {
		super(instancePipe, seeds, metric, proxy);
		this.sigma = sigma;

	}

	public Clustering cluster(InstanceList instances){

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
				double sum = 0.0;
				for(int k=0; k<fv_i.getValues().length; k++){
					sum += Math.pow((fv_i.getValues()[k]-fv_j.getValues()[k]), 2)/sigma;
				}
				weightMatrix[i][j] = Math.exp(-sum);
//				weightMatrix[i][j] = 1-metric.distance(fv_i, fv_j);
			}
		}
		int clusterLabels[] = new int[instances.size()];
		try {
						
			MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
			
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
			proxy.eval("L = D - A");
			proxy.eval("I = eye(size(L))");
			proxy.eval("F = mldivide((99*L+I),(Y))");
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
			for(int i=1; i<=10; i++){
				proxy.eval("A1 = A./(2*W)");
//				proxy.eval("A1 = A1.*(V<=epsilon)");
				proxy.eval("A1 = (A1+A1')/2");
				proxy.eval("d1 = sum(A1,2)");
				proxy.eval("D1 = diag(d1)");
				proxy.eval("L1 = D1 - A1");
				proxy.eval("F = mldivide((99*L1+I),(Y))");
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
				proxy.eval("obj("+i+") = 0.99*0.5*sum(sum((A.*W)))+0.01*trace((F-Y)'*(F-Y))");
			}
			
/*			double[][] obj = processor.getNumericArray("obj").getRealArray2D();
			double[] arr = new double[20];
			for(int i=0; i<processor.getNumericArray("obj").getLength(); i++)
				arr[i] = obj[i][0];
					
			
			ArrayList list= new ArrayList(); 
			list.add(new MLDouble("obj", arr, processor.getNumericArray("obj").getLength()));
			String matInputFile = "/home/peng/Downloads/visual/obj.mat";
			 try { 
				 new MatFileWriter(matInputFile, list); 
			} catch(IOException e) { // TODO Auto-generated catch block
				 e.printStackTrace(); 
			}*/
			
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
