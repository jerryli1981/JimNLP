package edu.pengli.nlp.conference.cikm2012.types;

import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;

public class Summary extends InstanceList {
	

	public Summary(){
		super(null);
	}
	  
	public int length(){
		int length = 0;
		for(Instance sent : this){
			length += ((String)sent.getSource()).split(" ").length;
		}
		
		return length;
	}

}
