package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.ArrayList;

import edu.smu.tspell.wordnet.NounSynset;
import edu.smu.tspell.wordnet.Synset;
import edu.smu.tspell.wordnet.SynsetType;
import edu.smu.tspell.wordnet.VerbSynset;
import edu.smu.tspell.wordnet.WordNetDatabase;

public class WordnetTagger {
	
	static WordNetDatabase database;
	static NounSynset nounSynset;
	static NounSynset[] nounHyponyms;
	static VerbSynset verbSynset;
	static VerbSynset[] verbHyponyms;
	
	public WordnetTagger(){
		System.setProperty("wordnet.database.dir", "../models/WordNet/WordNet-3.0/dict");
		database = WordNetDatabase.getFileInstance();
	}
	


	public static ArrayList<String> getNounTypes(String noun){
		
		ArrayList<String> ret = new ArrayList<String>();
		Synset[] synsets = database.getSynsets(noun, SynsetType.NOUN);
		for (int i = 0; i < synsets.length; i++) {
		    nounSynset = (NounSynset)(synsets[i]);
		    nounHyponyms = nounSynset.getHypernyms();
		    for(int j=0; j<nounHyponyms.length; j++){
		    	String[] wordForms = nounHyponyms[j].getWordForms();
		    	for(int k=0; k<wordForms.length; k++){
		    		System.out.println(noun +" is a kind of "+ wordForms[k]);
		    		ret.add(wordForms[k]);
		    	}	
		    }
		    
		} 
		return ret;
	}
	
	public static ArrayList<String> getVerbTypes(String noun){
		
		ArrayList<String> ret = new ArrayList<String>();
		Synset[] synsets = database.getSynsets(noun, SynsetType.VERB);
		for (int i = 0; i < synsets.length; i++) {
		    verbSynset = (VerbSynset)(synsets[i]);
		    verbHyponyms = verbSynset.getHypernyms();
		    for(int j=0; j<verbHyponyms.length; j++){
		    	String[] wordForms = verbHyponyms[j].getWordForms();
		    	for(int k=0; k<wordForms.length; k++){
		    		System.out.println(noun +" is a kind of "+ wordForms[k]);
		    		ret.add(wordForms[k]);
		    	}	
		    } 
		} 
		return ret;
	}
	
	public static void main(String[] args){
		
		WordnetTagger obj = new WordnetTagger();
		System.out.println(obj.getNounTypes("beef"));
	}
}
