package edu.pengli.nlp.platform.pipe;

import java.util.Properties;

import edu.pengli.nlp.platform.types.Instance;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class CharSequenceTokenizationAndSentencesplit extends Pipe {

	AnnotationPipeline pipeline;

	public CharSequenceTokenizationAndSentencesplit() {

		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit");
		pipeline = new StanfordCoreNLP(props);

	}

	public Instance pipe(Instance inst) {
		String text = (String) inst.getData();
		Annotation document = new Annotation(text);
		pipeline.annotate(document);
		inst.setData(document);		
		return inst;
	}

}
