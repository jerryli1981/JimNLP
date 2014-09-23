package edu.pengli.nlp.platform.pipe;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import edu.pengli.nlp.platform.types.Instance;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.InvalidFormatException;

public class SentenceDetector extends Pipe{

	SentenceDetectorME sd;
	
	public SentenceDetector(){
		InputStream modelIn = null;
		SentenceModel model;

		try {
			modelIn = new FileInputStream("models/OpenNLP/en-sent.bin");
			model = new SentenceModel(modelIn);
			sd = new SentenceDetectorME(model);
		} catch (FileNotFoundException e) {

			e.printStackTrace();
		} catch (InvalidFormatException e) {

			e.printStackTrace();
		} catch (IOException e) {

			e.printStackTrace();
		} finally {
			if (modelIn != null) {
				try {
					modelIn.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public Instance pipe(Instance inst) {
		String[] sents = sd.sentDetect((String) inst.getData());
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<sents.length; i++){
			sb.append(sents[i]+"\n");
		}
		inst.setData(sb.toString().trim());
		return inst;
	}
	
}
