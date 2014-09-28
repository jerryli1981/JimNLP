package edu.pengli.nlp.conference.acl2015.pipe;


import java.util.regex.Matcher;

import edu.pengli.nlp.platform.pipe.CharSequenceRemoveHTML;
import edu.pengli.nlp.platform.types.Instance;

public class CharSequenceExtractContent extends CharSequenceRemoveHTML {
	
	public CharSequenceExtractContent(String regex){
		super(regex);
	}
	public Instance pipe(Instance carrier) {
		
		String content = carrier.getData().toString();
        Matcher m = p.matcher(content);
        if(m.find()){
        	content = m.group();
        	content = content.replaceAll("</s>PARA", "</s>");
        	content = content.replaceAll("<.*?>", "");
        	content = content.replaceAll("\"", "");
        }
		carrier.setData((CharSequence) content.trim());
		return carrier;
	}

}
