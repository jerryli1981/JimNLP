package edu.pengli.nlp.platform.pipe;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import edu.pengli.nlp.platform.types.Instance;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

public class SentenceTokenization extends Pipe {

	Tokenizer tokenizer;

	public SentenceTokenization() {
		InputStream modelIn = null;
		TokenizerModel model;

		try {
			modelIn = new FileInputStream("../models/OpenNLP/en-token.bin");
			model = new TokenizerModel(modelIn);
			tokenizer = new TokenizerME(model);
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
		String input = (String) inst.getData();
		String[] sents = input.split("\n");
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<sents.length; i++){
			StringBuilder sbb = new StringBuilder();
			String[] tokens = tokenizer.tokenize(sents[i]);
			for(int j=0; j<tokens.length; j++){
				if(tokens[j].matches("\\p{Alpha}+|\\p{Digit}+|\\p{Punct}"));
				sbb.append(tokens[j]+" ");
			}
		    sb.append(sbb.toString().trim()+"\n");
		}

		inst.setData(sb.toString().trim());
		return inst;
	}

}
