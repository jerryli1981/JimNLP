package edu.pengli.nlp.platform.pipe;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.TreeSet;

import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.Token;
import edu.pengli.nlp.platform.types.TokenSequence;


public class TokenSequenceRemoveStopwords extends Pipe{
	
	TreeSet<String> stopWordsDict;
	
	public TokenSequenceRemoveStopwords(){
		stopWordsDict = new TreeSet<String>();
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader("models/stop_words.dat"));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			for(String str=reader.readLine();str!=null;str=reader.readLine())
			{
				stopWordsDict.add(str);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public Instance pipe (Instance carrier)
	{
		TokenSequence ts = (TokenSequence) carrier.getData();
		TokenSequence ret = new TokenSequence ();

		for (int i = 0; i < ts.size(); i++) {
			Token t = ts.get(i);
			if (! stopWordsDict.contains (t.getMention().toLowerCase())) {
				ret.add (t);
			}
		}
		carrier.setData(ret);
		return carrier;
	}

}
