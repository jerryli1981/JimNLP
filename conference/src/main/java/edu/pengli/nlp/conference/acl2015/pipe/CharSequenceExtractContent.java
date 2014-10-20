package edu.pengli.nlp.conference.acl2015.pipe;


import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.pengli.nlp.platform.pipe.CharSequenceRemoveHTML;
import edu.pengli.nlp.platform.types.Instance;

public class CharSequenceExtractContent extends CharSequenceRemoveHTML {
	
	public CharSequenceExtractContent(String regex){
		super(regex);
	}
	public Instance pipe(Instance carrier) {
		//<s docid="APW_ENG_20070615.0356" num="2" stype="1">FORT LAUDERDALE, Florida 2007-06-15 05:06:17 UTC</s>PARA
		String content = carrier.getData().toString();
       
        Pattern fil = Pattern.compile(">.*\\s(UTC)");
        Matcher mfil = fil.matcher(content);
        if(mfil.find()){
        	String find = mfil.group();
        	content = content.replace(find, ">");
        }
        
        Matcher m = p.matcher(content);
        if(m.find()){
        	content = m.group();
        	content = content.replaceAll("</s>PARA", "</s>");
        	content = content.replaceAll("<.*?>", "");
        	content = content.replaceAll("\"", "");
        	content = content.replaceAll("\\s?\\(.*?\\)\\s?", " ");
        	content = content.replaceAll("'ll", " will");
        	content = content.replaceAll("``", "");
        }
		carrier.setData((CharSequence) content.trim());
		return carrier;
	}

}
