package edu.pengli.nlp.platform.algorithms.classify;

import java.io.IOException;
import java.util.ArrayList;


import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;

public abstract class SemiSupervisedClustering extends Clusterer {
	
	MatlabProxy proxy;
	InstanceList seeds;
	Metric metric;

	public SemiSupervisedClustering(Pipe instancePipe, InstanceList seeds, Metric metric,
			MatlabProxy proxy) {

		super(instancePipe);

		this.metric = metric;
		this.seeds = seeds;
		this.proxy = proxy;

	}

}
