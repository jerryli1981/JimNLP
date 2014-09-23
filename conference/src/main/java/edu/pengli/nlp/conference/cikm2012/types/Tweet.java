package edu.pengli.nlp.conference.cikm2012.types;

import java.io.Serializable;

import edu.pengli.nlp.platform.types.Sentence;

public class Tweet extends Sentence implements Serializable{

	private String id;
	private String mention;
	private int score;

}
