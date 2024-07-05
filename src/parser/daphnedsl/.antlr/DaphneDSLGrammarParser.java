// Generated from /home/niklas/daphne/src/parser/daphnedsl/DaphneDSLGrammar.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue"})
public class DaphneDSLGrammarParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, T__6=7, T__7=8, T__8=9, 
		T__9=10, T__10=11, T__11=12, T__12=13, T__13=14, T__14=15, T__15=16, T__16=17, 
		T__17=18, T__18=19, T__19=20, T__20=21, T__21=22, T__22=23, T__23=24, 
		T__24=25, T__25=26, T__26=27, T__27=28, T__28=29, T__29=30, T__30=31, 
		T__31=32, KW_IF=33, KW_ELSE=34, KW_WHILE=35, KW_DO=36, KW_FOR=37, KW_IN=38, 
		KW_TRUE=39, KW_FALSE=40, KW_AS=41, KW_DEF=42, KW_RETURN=43, KW_IMPORT=44, 
		DATA_TYPE=45, VALUE_TYPE=46, INT_LITERAL=47, FLOAT_LITERAL=48, STRING_LITERAL=49, 
		IDENTIFIER=50, SCRIPT_STYLE_LINE_COMMENT=51, C_STYLE_LINE_COMMENT=52, 
		MULTILINE_BLOCK_COMMENT=53, WS=54;
	public static final int
		RULE_script = 0, RULE_statement = 1, RULE_importStatement = 2, RULE_blockStatement = 3, 
		RULE_exprStatement = 4, RULE_assignStatement = 5, RULE_ifStatement = 6, 
		RULE_whileStatement = 7, RULE_forStatement = 8, RULE_functionStatement = 9, 
		RULE_returnStatement = 10, RULE_functionArgs = 11, RULE_functionArg = 12, 
		RULE_functionRetTypes = 13, RULE_funcTypeDef = 14, RULE_expr = 15, RULE_frameRow = 16, 
		RULE_indexing = 17, RULE_range = 18, RULE_literal = 19, RULE_boolLiteral = 20;
	private static String[] makeRuleNames() {
		return new String[] {
			"script", "statement", "importStatement", "blockStatement", "exprStatement", 
			"assignStatement", "ifStatement", "whileStatement", "forStatement", "functionStatement", 
			"returnStatement", "functionArgs", "functionArg", "functionRetTypes", 
			"funcTypeDef", "expr", "frameRow", "indexing", "range", "literal", "boolLiteral"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "';'", "'{'", "'}'", "'.'", "','", "'='", "'('", "')'", "':'", 
			"'->'", "'<'", "'>'", "'$'", "'::'", "'[['", "']]'", "'@'", "'^'", "'%'", 
			"'*'", "'/'", "'+'", "'-'", "'=='", "'!='", "'<='", "'>='", "'&&'", "'||'", 
			"'?'", "'['", "']'", "'if'", "'else'", "'while'", "'do'", "'for'", "'in'", 
			"'true'", "'false'", "'as'", "'def'", "'return'", "'import'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, "KW_IF", "KW_ELSE", 
			"KW_WHILE", "KW_DO", "KW_FOR", "KW_IN", "KW_TRUE", "KW_FALSE", "KW_AS", 
			"KW_DEF", "KW_RETURN", "KW_IMPORT", "DATA_TYPE", "VALUE_TYPE", "INT_LITERAL", 
			"FLOAT_LITERAL", "STRING_LITERAL", "IDENTIFIER", "SCRIPT_STYLE_LINE_COMMENT", 
			"C_STYLE_LINE_COMMENT", "MULTILINE_BLOCK_COMMENT", "WS"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "DaphneDSLGrammar.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public DaphneDSLGrammarParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ScriptContext extends ParserRuleContext {
		public TerminalNode EOF() { return getToken(DaphneDSLGrammarParser.EOF, 0); }
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public ScriptContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_script; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterScript(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitScript(this);
		}
	}

	public final ScriptContext script() throws RecognitionException {
		ScriptContext _localctx = new ScriptContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_script);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(45);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2145948197200004L) != 0)) {
				{
				{
				setState(42);
				statement();
				}
				}
				setState(47);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(48);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StatementContext extends ParserRuleContext {
		public BlockStatementContext blockStatement() {
			return getRuleContext(BlockStatementContext.class,0);
		}
		public ExprStatementContext exprStatement() {
			return getRuleContext(ExprStatementContext.class,0);
		}
		public AssignStatementContext assignStatement() {
			return getRuleContext(AssignStatementContext.class,0);
		}
		public IfStatementContext ifStatement() {
			return getRuleContext(IfStatementContext.class,0);
		}
		public WhileStatementContext whileStatement() {
			return getRuleContext(WhileStatementContext.class,0);
		}
		public ForStatementContext forStatement() {
			return getRuleContext(ForStatementContext.class,0);
		}
		public FunctionStatementContext functionStatement() {
			return getRuleContext(FunctionStatementContext.class,0);
		}
		public ReturnStatementContext returnStatement() {
			return getRuleContext(ReturnStatementContext.class,0);
		}
		public ImportStatementContext importStatement() {
			return getRuleContext(ImportStatementContext.class,0);
		}
		public StatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_statement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitStatement(this);
		}
	}

	public final StatementContext statement() throws RecognitionException {
		StatementContext _localctx = new StatementContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_statement);
		try {
			setState(59);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,1,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(50);
				blockStatement();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(51);
				exprStatement();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(52);
				assignStatement();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(53);
				ifStatement();
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(54);
				whileStatement();
				}
				break;
			case 6:
				enterOuterAlt(_localctx, 6);
				{
				setState(55);
				forStatement();
				}
				break;
			case 7:
				enterOuterAlt(_localctx, 7);
				{
				setState(56);
				functionStatement();
				}
				break;
			case 8:
				enterOuterAlt(_localctx, 8);
				{
				setState(57);
				returnStatement();
				}
				break;
			case 9:
				enterOuterAlt(_localctx, 9);
				{
				setState(58);
				importStatement();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ImportStatementContext extends ParserRuleContext {
		public Token filePath;
		public Token alias;
		public TerminalNode KW_IMPORT() { return getToken(DaphneDSLGrammarParser.KW_IMPORT, 0); }
		public List<TerminalNode> STRING_LITERAL() { return getTokens(DaphneDSLGrammarParser.STRING_LITERAL); }
		public TerminalNode STRING_LITERAL(int i) {
			return getToken(DaphneDSLGrammarParser.STRING_LITERAL, i);
		}
		public TerminalNode KW_AS() { return getToken(DaphneDSLGrammarParser.KW_AS, 0); }
		public ImportStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_importStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterImportStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitImportStatement(this);
		}
	}

	public final ImportStatementContext importStatement() throws RecognitionException {
		ImportStatementContext _localctx = new ImportStatementContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_importStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(61);
			match(KW_IMPORT);
			setState(62);
			((ImportStatementContext)_localctx).filePath = match(STRING_LITERAL);
			setState(65);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==KW_AS) {
				{
				setState(63);
				match(KW_AS);
				setState(64);
				((ImportStatementContext)_localctx).alias = match(STRING_LITERAL);
				}
			}

			setState(67);
			match(T__0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class BlockStatementContext extends ParserRuleContext {
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public BlockStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_blockStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterBlockStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitBlockStatement(this);
		}
	}

	public final BlockStatementContext blockStatement() throws RecognitionException {
		BlockStatementContext _localctx = new BlockStatementContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_blockStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(69);
			match(T__1);
			setState(73);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2145948197200004L) != 0)) {
				{
				{
				setState(70);
				statement();
				}
				}
				setState(75);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(76);
			match(T__2);
			setState(78);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__0) {
				{
				setState(77);
				match(T__0);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExprStatementContext extends ParserRuleContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ExprStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_exprStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterExprStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitExprStatement(this);
		}
	}

	public final ExprStatementContext exprStatement() throws RecognitionException {
		ExprStatementContext _localctx = new ExprStatementContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_exprStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(80);
			expr(0);
			setState(81);
			match(T__0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssignStatementContext extends ParserRuleContext {
		public List<TerminalNode> IDENTIFIER() { return getTokens(DaphneDSLGrammarParser.IDENTIFIER); }
		public TerminalNode IDENTIFIER(int i) {
			return getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
		}
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public List<IndexingContext> indexing() {
			return getRuleContexts(IndexingContext.class);
		}
		public IndexingContext indexing(int i) {
			return getRuleContext(IndexingContext.class,i);
		}
		public AssignStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assignStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterAssignStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitAssignStatement(this);
		}
	}

	public final AssignStatementContext assignStatement() throws RecognitionException {
		AssignStatementContext _localctx = new AssignStatementContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_assignStatement);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(87);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,5,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(83);
					match(IDENTIFIER);
					setState(84);
					match(T__3);
					}
					} 
				}
				setState(89);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,5,_ctx);
			}
			setState(90);
			match(IDENTIFIER);
			setState(92);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__30) {
				{
				setState(91);
				indexing();
				}
			}

			setState(108);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__4) {
				{
				{
				setState(94);
				match(T__4);
				setState(99);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
				while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
					if ( _alt==1 ) {
						{
						{
						setState(95);
						match(IDENTIFIER);
						setState(96);
						match(T__3);
						}
						} 
					}
					setState(101);
					_errHandler.sync(this);
					_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
				}
				setState(102);
				match(IDENTIFIER);
				setState(104);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__30) {
					{
					setState(103);
					indexing();
					}
				}

				}
				}
				setState(110);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(111);
			match(T__5);
			setState(112);
			expr(0);
			setState(113);
			match(T__0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IfStatementContext extends ParserRuleContext {
		public ExprContext cond;
		public StatementContext thenStmt;
		public StatementContext elseStmt;
		public TerminalNode KW_IF() { return getToken(DaphneDSLGrammarParser.KW_IF, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public TerminalNode KW_ELSE() { return getToken(DaphneDSLGrammarParser.KW_ELSE, 0); }
		public IfStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_ifStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterIfStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitIfStatement(this);
		}
	}

	public final IfStatementContext ifStatement() throws RecognitionException {
		IfStatementContext _localctx = new IfStatementContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_ifStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(115);
			match(KW_IF);
			setState(116);
			match(T__6);
			setState(117);
			((IfStatementContext)_localctx).cond = expr(0);
			setState(118);
			match(T__7);
			setState(119);
			((IfStatementContext)_localctx).thenStmt = statement();
			setState(122);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,10,_ctx) ) {
			case 1:
				{
				setState(120);
				match(KW_ELSE);
				setState(121);
				((IfStatementContext)_localctx).elseStmt = statement();
				}
				break;
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class WhileStatementContext extends ParserRuleContext {
		public ExprContext cond;
		public StatementContext bodyStmt;
		public TerminalNode KW_WHILE() { return getToken(DaphneDSLGrammarParser.KW_WHILE, 0); }
		public TerminalNode KW_DO() { return getToken(DaphneDSLGrammarParser.KW_DO, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public StatementContext statement() {
			return getRuleContext(StatementContext.class,0);
		}
		public WhileStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_whileStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterWhileStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitWhileStatement(this);
		}
	}

	public final WhileStatementContext whileStatement() throws RecognitionException {
		WhileStatementContext _localctx = new WhileStatementContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_whileStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(139);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case KW_WHILE:
				{
				setState(124);
				match(KW_WHILE);
				setState(125);
				match(T__6);
				setState(126);
				((WhileStatementContext)_localctx).cond = expr(0);
				setState(127);
				match(T__7);
				setState(128);
				((WhileStatementContext)_localctx).bodyStmt = statement();
				}
				break;
			case KW_DO:
				{
				setState(130);
				match(KW_DO);
				setState(131);
				((WhileStatementContext)_localctx).bodyStmt = statement();
				setState(132);
				match(KW_WHILE);
				setState(133);
				match(T__6);
				setState(134);
				((WhileStatementContext)_localctx).cond = expr(0);
				setState(135);
				match(T__7);
				setState(137);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__0) {
					{
					setState(136);
					match(T__0);
					}
				}

				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ForStatementContext extends ParserRuleContext {
		public Token var;
		public ExprContext from;
		public ExprContext to;
		public ExprContext step;
		public StatementContext bodyStmt;
		public TerminalNode KW_FOR() { return getToken(DaphneDSLGrammarParser.KW_FOR, 0); }
		public TerminalNode KW_IN() { return getToken(DaphneDSLGrammarParser.KW_IN, 0); }
		public TerminalNode IDENTIFIER() { return getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public StatementContext statement() {
			return getRuleContext(StatementContext.class,0);
		}
		public ForStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_forStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterForStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitForStatement(this);
		}
	}

	public final ForStatementContext forStatement() throws RecognitionException {
		ForStatementContext _localctx = new ForStatementContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_forStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(141);
			match(KW_FOR);
			setState(142);
			match(T__6);
			setState(143);
			((ForStatementContext)_localctx).var = match(IDENTIFIER);
			setState(144);
			match(KW_IN);
			setState(145);
			((ForStatementContext)_localctx).from = expr(0);
			setState(146);
			match(T__8);
			setState(147);
			((ForStatementContext)_localctx).to = expr(0);
			setState(150);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__8) {
				{
				setState(148);
				match(T__8);
				setState(149);
				((ForStatementContext)_localctx).step = expr(0);
				}
			}

			setState(152);
			match(T__7);
			setState(153);
			((ForStatementContext)_localctx).bodyStmt = statement();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionStatementContext extends ParserRuleContext {
		public Token name;
		public FunctionArgsContext args;
		public FunctionRetTypesContext retTys;
		public BlockStatementContext bodyStmt;
		public TerminalNode KW_DEF() { return getToken(DaphneDSLGrammarParser.KW_DEF, 0); }
		public TerminalNode IDENTIFIER() { return getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
		public BlockStatementContext blockStatement() {
			return getRuleContext(BlockStatementContext.class,0);
		}
		public FunctionArgsContext functionArgs() {
			return getRuleContext(FunctionArgsContext.class,0);
		}
		public FunctionRetTypesContext functionRetTypes() {
			return getRuleContext(FunctionRetTypesContext.class,0);
		}
		public FunctionStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterFunctionStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitFunctionStatement(this);
		}
	}

	public final FunctionStatementContext functionStatement() throws RecognitionException {
		FunctionStatementContext _localctx = new FunctionStatementContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_functionStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(155);
			match(KW_DEF);
			setState(156);
			((FunctionStatementContext)_localctx).name = match(IDENTIFIER);
			setState(157);
			match(T__6);
			setState(159);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==IDENTIFIER) {
				{
				setState(158);
				((FunctionStatementContext)_localctx).args = functionArgs();
				}
			}

			setState(161);
			match(T__7);
			setState(164);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__9) {
				{
				setState(162);
				match(T__9);
				setState(163);
				((FunctionStatementContext)_localctx).retTys = functionRetTypes();
				}
			}

			setState(166);
			((FunctionStatementContext)_localctx).bodyStmt = blockStatement();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ReturnStatementContext extends ParserRuleContext {
		public TerminalNode KW_RETURN() { return getToken(DaphneDSLGrammarParser.KW_RETURN, 0); }
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public ReturnStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_returnStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterReturnStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitReturnStatement(this);
		}
	}

	public final ReturnStatementContext returnStatement() throws RecognitionException {
		ReturnStatementContext _localctx = new ReturnStatementContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_returnStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(168);
			match(KW_RETURN);
			setState(177);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
				{
				setState(169);
				expr(0);
				setState(174);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__4) {
					{
					{
					setState(170);
					match(T__4);
					setState(171);
					expr(0);
					}
					}
					setState(176);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(179);
			match(T__0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionArgsContext extends ParserRuleContext {
		public List<FunctionArgContext> functionArg() {
			return getRuleContexts(FunctionArgContext.class);
		}
		public FunctionArgContext functionArg(int i) {
			return getRuleContext(FunctionArgContext.class,i);
		}
		public FunctionArgsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionArgs; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterFunctionArgs(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitFunctionArgs(this);
		}
	}

	public final FunctionArgsContext functionArgs() throws RecognitionException {
		FunctionArgsContext _localctx = new FunctionArgsContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_functionArgs);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(181);
			functionArg();
			setState(186);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,18,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(182);
					match(T__4);
					setState(183);
					functionArg();
					}
					} 
				}
				setState(188);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,18,_ctx);
			}
			setState(190);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__4) {
				{
				setState(189);
				match(T__4);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionArgContext extends ParserRuleContext {
		public Token var;
		public FuncTypeDefContext ty;
		public TerminalNode IDENTIFIER() { return getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
		public FuncTypeDefContext funcTypeDef() {
			return getRuleContext(FuncTypeDefContext.class,0);
		}
		public FunctionArgContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionArg; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterFunctionArg(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitFunctionArg(this);
		}
	}

	public final FunctionArgContext functionArg() throws RecognitionException {
		FunctionArgContext _localctx = new FunctionArgContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_functionArg);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(192);
			((FunctionArgContext)_localctx).var = match(IDENTIFIER);
			setState(195);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__8) {
				{
				setState(193);
				match(T__8);
				setState(194);
				((FunctionArgContext)_localctx).ty = funcTypeDef();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionRetTypesContext extends ParserRuleContext {
		public List<FuncTypeDefContext> funcTypeDef() {
			return getRuleContexts(FuncTypeDefContext.class);
		}
		public FuncTypeDefContext funcTypeDef(int i) {
			return getRuleContext(FuncTypeDefContext.class,i);
		}
		public FunctionRetTypesContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionRetTypes; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterFunctionRetTypes(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitFunctionRetTypes(this);
		}
	}

	public final FunctionRetTypesContext functionRetTypes() throws RecognitionException {
		FunctionRetTypesContext _localctx = new FunctionRetTypesContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_functionRetTypes);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(197);
			funcTypeDef();
			setState(202);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__4) {
				{
				{
				setState(198);
				match(T__4);
				setState(199);
				funcTypeDef();
				}
				}
				setState(204);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FuncTypeDefContext extends ParserRuleContext {
		public Token dataTy;
		public Token elTy;
		public Token scalarTy;
		public TerminalNode DATA_TYPE() { return getToken(DaphneDSLGrammarParser.DATA_TYPE, 0); }
		public TerminalNode VALUE_TYPE() { return getToken(DaphneDSLGrammarParser.VALUE_TYPE, 0); }
		public FuncTypeDefContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_funcTypeDef; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterFuncTypeDef(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitFuncTypeDef(this);
		}
	}

	public final FuncTypeDefContext funcTypeDef() throws RecognitionException {
		FuncTypeDefContext _localctx = new FuncTypeDefContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_funcTypeDef);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(212);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case DATA_TYPE:
				{
				setState(205);
				((FuncTypeDefContext)_localctx).dataTy = match(DATA_TYPE);
				setState(209);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__10) {
					{
					setState(206);
					match(T__10);
					setState(207);
					((FuncTypeDefContext)_localctx).elTy = match(VALUE_TYPE);
					setState(208);
					match(T__11);
					}
				}

				}
				break;
			case VALUE_TYPE:
				{
				setState(211);
				((FuncTypeDefContext)_localctx).scalarTy = match(VALUE_TYPE);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExprContext extends ParserRuleContext {
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
	 
		public ExprContext() { }
		public void copyFrom(ExprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class RightIdxExtractExprContext extends ExprContext {
		public ExprContext obj;
		public IndexingContext idx;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public IndexingContext indexing() {
			return getRuleContext(IndexingContext.class,0);
		}
		public RightIdxExtractExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterRightIdxExtractExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitRightIdxExtractExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ModExprContext extends ExprContext {
		public ExprContext lhs;
		public Token op;
		public ExprContext rhs;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public ModExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterModExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitModExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class CastExprContext extends ExprContext {
		public TerminalNode KW_AS() { return getToken(DaphneDSLGrammarParser.KW_AS, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode DATA_TYPE() { return getToken(DaphneDSLGrammarParser.DATA_TYPE, 0); }
		public TerminalNode VALUE_TYPE() { return getToken(DaphneDSLGrammarParser.VALUE_TYPE, 0); }
		public CastExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterCastExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitCastExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class MatmulExprContext extends ExprContext {
		public ExprContext lhs;
		public Token op;
		public ExprContext rhs;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public MatmulExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterMatmulExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitMatmulExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class CondExprContext extends ExprContext {
		public ExprContext cond;
		public ExprContext thenExpr;
		public ExprContext elseExpr;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public CondExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterCondExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitCondExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ConjExprContext extends ExprContext {
		public ExprContext lhs;
		public Token op;
		public ExprContext rhs;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public ConjExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterConjExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitConjExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class DisjExprContext extends ExprContext {
		public ExprContext lhs;
		public Token op;
		public ExprContext rhs;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public DisjExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterDisjExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitDisjExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class RightIdxFilterExprContext extends ExprContext {
		public ExprContext obj;
		public ExprContext rows;
		public ExprContext cols;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public RightIdxFilterExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterRightIdxFilterExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitRightIdxFilterExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class MatrixLiteralExprContext extends ExprContext {
		public ExprContext rows;
		public ExprContext cols;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public MatrixLiteralExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterMatrixLiteralExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitMatrixLiteralExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ParanthesesExprContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ParanthesesExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterParanthesesExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitParanthesesExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class CmpExprContext extends ExprContext {
		public ExprContext lhs;
		public Token op;
		public ExprContext rhs;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public CmpExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterCmpExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitCmpExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class AddExprContext extends ExprContext {
		public ExprContext lhs;
		public Token op;
		public ExprContext rhs;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public AddExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterAddExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitAddExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class LiteralExprContext extends ExprContext {
		public LiteralContext literal() {
			return getRuleContext(LiteralContext.class,0);
		}
		public LiteralExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterLiteralExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitLiteralExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class RowMajorFrameLiteralExprContext extends ExprContext {
		public FrameRowContext labels;
		public FrameRowContext frameRow;
		public List<FrameRowContext> rows = new ArrayList<FrameRowContext>();
		public List<FrameRowContext> frameRow() {
			return getRuleContexts(FrameRowContext.class);
		}
		public FrameRowContext frameRow(int i) {
			return getRuleContext(FrameRowContext.class,i);
		}
		public RowMajorFrameLiteralExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterRowMajorFrameLiteralExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitRowMajorFrameLiteralExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class MulExprContext extends ExprContext {
		public ExprContext lhs;
		public Token op;
		public ExprContext rhs;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public MulExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterMulExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitMulExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ArgExprContext extends ExprContext {
		public Token arg;
		public TerminalNode IDENTIFIER() { return getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
		public ArgExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterArgExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitArgExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ColMajorFrameLiteralExprContext extends ExprContext {
		public ExprContext expr;
		public List<ExprContext> labels = new ArrayList<ExprContext>();
		public List<ExprContext> cols = new ArrayList<ExprContext>();
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public ColMajorFrameLiteralExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterColMajorFrameLiteralExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitColMajorFrameLiteralExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class CallExprContext extends ExprContext {
		public Token ns;
		public Token func;
		public Token kernel;
		public List<TerminalNode> IDENTIFIER() { return getTokens(DaphneDSLGrammarParser.IDENTIFIER); }
		public TerminalNode IDENTIFIER(int i) {
			return getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
		}
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public CallExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterCallExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitCallExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class PowExprContext extends ExprContext {
		public ExprContext lhs;
		public Token op;
		public ExprContext rhs;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public PowExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterPowExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitPowExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class IdentifierExprContext extends ExprContext {
		public List<TerminalNode> IDENTIFIER() { return getTokens(DaphneDSLGrammarParser.IDENTIFIER); }
		public TerminalNode IDENTIFIER(int i) {
			return getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
		}
		public IdentifierExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterIdentifierExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitIdentifierExpr(this);
		}
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 30;
		enterRecursionRule(_localctx, 30, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(321);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,38,_ctx) ) {
			case 1:
				{
				_localctx = new LiteralExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(215);
				literal();
				}
				break;
			case 2:
				{
				_localctx = new ArgExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(216);
				match(T__12);
				setState(217);
				((ArgExprContext)_localctx).arg = match(IDENTIFIER);
				}
				break;
			case 3:
				{
				_localctx = new IdentifierExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				{
				setState(222);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,24,_ctx);
				while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
					if ( _alt==1 ) {
						{
						{
						setState(218);
						match(IDENTIFIER);
						setState(219);
						match(T__3);
						}
						} 
					}
					setState(224);
					_errHandler.sync(this);
					_alt = getInterpreter().adaptivePredict(_input,24,_ctx);
				}
				setState(225);
				match(IDENTIFIER);
				}
				}
				break;
			case 4:
				{
				_localctx = new ParanthesesExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(226);
				match(T__6);
				setState(227);
				expr(0);
				setState(228);
				match(T__7);
				}
				break;
			case 5:
				{
				_localctx = new CallExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(234);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,25,_ctx);
				while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
					if ( _alt==1 ) {
						{
						{
						setState(230);
						((CallExprContext)_localctx).ns = match(IDENTIFIER);
						setState(231);
						match(T__3);
						}
						} 
					}
					setState(236);
					_errHandler.sync(this);
					_alt = getInterpreter().adaptivePredict(_input,25,_ctx);
				}
				setState(237);
				((CallExprContext)_localctx).func = match(IDENTIFIER);
				setState(240);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__13) {
					{
					setState(238);
					match(T__13);
					setState(239);
					((CallExprContext)_localctx).kernel = match(IDENTIFIER);
					}
				}

				setState(242);
				match(T__6);
				setState(251);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
					{
					setState(243);
					expr(0);
					setState(248);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__4) {
						{
						{
						setState(244);
						match(T__4);
						setState(245);
						expr(0);
						}
						}
						setState(250);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				setState(253);
				match(T__7);
				}
				break;
			case 6:
				{
				_localctx = new CastExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(254);
				match(KW_AS);
				setState(264);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,29,_ctx) ) {
				case 1:
					{
					{
					setState(255);
					match(T__3);
					setState(256);
					match(DATA_TYPE);
					}
					}
					break;
				case 2:
					{
					{
					setState(257);
					match(T__3);
					setState(258);
					match(VALUE_TYPE);
					}
					}
					break;
				case 3:
					{
					{
					setState(259);
					match(T__3);
					setState(260);
					match(DATA_TYPE);
					setState(261);
					match(T__10);
					setState(262);
					match(VALUE_TYPE);
					setState(263);
					match(T__11);
					}
					}
					break;
				}
				setState(266);
				match(T__6);
				setState(267);
				expr(0);
				setState(268);
				match(T__7);
				}
				break;
			case 7:
				{
				_localctx = new MatrixLiteralExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(270);
				match(T__30);
				setState(279);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
					{
					setState(271);
					expr(0);
					setState(276);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__4) {
						{
						{
						setState(272);
						match(T__4);
						setState(273);
						expr(0);
						}
						}
						setState(278);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				setState(281);
				match(T__31);
				setState(291);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,34,_ctx) ) {
				case 1:
					{
					setState(282);
					match(T__6);
					setState(284);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
						{
						setState(283);
						((MatrixLiteralExprContext)_localctx).rows = expr(0);
						}
					}

					setState(286);
					match(T__4);
					setState(288);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
						{
						setState(287);
						((MatrixLiteralExprContext)_localctx).cols = expr(0);
						}
					}

					setState(290);
					match(T__7);
					}
					break;
				}
				}
				break;
			case 8:
				{
				_localctx = new ColMajorFrameLiteralExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(293);
				match(T__1);
				setState(307);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
					{
					setState(294);
					((ColMajorFrameLiteralExprContext)_localctx).expr = expr(0);
					((ColMajorFrameLiteralExprContext)_localctx).labels.add(((ColMajorFrameLiteralExprContext)_localctx).expr);
					setState(295);
					match(T__8);
					setState(296);
					((ColMajorFrameLiteralExprContext)_localctx).expr = expr(0);
					((ColMajorFrameLiteralExprContext)_localctx).cols.add(((ColMajorFrameLiteralExprContext)_localctx).expr);
					setState(304);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__4) {
						{
						{
						setState(297);
						match(T__4);
						setState(298);
						((ColMajorFrameLiteralExprContext)_localctx).expr = expr(0);
						((ColMajorFrameLiteralExprContext)_localctx).labels.add(((ColMajorFrameLiteralExprContext)_localctx).expr);
						setState(299);
						match(T__8);
						setState(300);
						((ColMajorFrameLiteralExprContext)_localctx).expr = expr(0);
						((ColMajorFrameLiteralExprContext)_localctx).cols.add(((ColMajorFrameLiteralExprContext)_localctx).expr);
						}
						}
						setState(306);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				setState(309);
				match(T__2);
				}
				break;
			case 9:
				{
				_localctx = new RowMajorFrameLiteralExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(310);
				match(T__1);
				setState(311);
				((RowMajorFrameLiteralExprContext)_localctx).labels = frameRow();
				setState(316);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__4) {
					{
					{
					setState(312);
					match(T__4);
					setState(313);
					((RowMajorFrameLiteralExprContext)_localctx).frameRow = frameRow();
					((RowMajorFrameLiteralExprContext)_localctx).rows.add(((RowMajorFrameLiteralExprContext)_localctx).frameRow);
					}
					}
					setState(318);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(319);
				match(T__2);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(367);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,42,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(365);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,41,_ctx) ) {
					case 1:
						{
						_localctx = new MatmulExprContext(new ExprContext(_parentctx, _parentState));
						((MatmulExprContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(323);
						if (!(precpred(_ctx, 12))) throw new FailedPredicateException(this, "precpred(_ctx, 12)");
						setState(324);
						((MatmulExprContext)_localctx).op = match(T__16);
						setState(325);
						((MatmulExprContext)_localctx).rhs = expr(13);
						}
						break;
					case 2:
						{
						_localctx = new PowExprContext(new ExprContext(_parentctx, _parentState));
						((PowExprContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(326);
						if (!(precpred(_ctx, 11))) throw new FailedPredicateException(this, "precpred(_ctx, 11)");
						setState(327);
						((PowExprContext)_localctx).op = match(T__17);
						setState(328);
						((PowExprContext)_localctx).rhs = expr(12);
						}
						break;
					case 3:
						{
						_localctx = new ModExprContext(new ExprContext(_parentctx, _parentState));
						((ModExprContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(329);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(330);
						((ModExprContext)_localctx).op = match(T__18);
						setState(331);
						((ModExprContext)_localctx).rhs = expr(11);
						}
						break;
					case 4:
						{
						_localctx = new MulExprContext(new ExprContext(_parentctx, _parentState));
						((MulExprContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(332);
						if (!(precpred(_ctx, 9))) throw new FailedPredicateException(this, "precpred(_ctx, 9)");
						setState(333);
						((MulExprContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==T__19 || _la==T__20) ) {
							((MulExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(334);
						((MulExprContext)_localctx).rhs = expr(10);
						}
						break;
					case 5:
						{
						_localctx = new AddExprContext(new ExprContext(_parentctx, _parentState));
						((AddExprContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(335);
						if (!(precpred(_ctx, 8))) throw new FailedPredicateException(this, "precpred(_ctx, 8)");
						setState(336);
						((AddExprContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==T__21 || _la==T__22) ) {
							((AddExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(337);
						((AddExprContext)_localctx).rhs = expr(9);
						}
						break;
					case 6:
						{
						_localctx = new CmpExprContext(new ExprContext(_parentctx, _parentState));
						((CmpExprContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(338);
						if (!(precpred(_ctx, 7))) throw new FailedPredicateException(this, "precpred(_ctx, 7)");
						setState(339);
						((CmpExprContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 251664384L) != 0)) ) {
							((CmpExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(340);
						((CmpExprContext)_localctx).rhs = expr(8);
						}
						break;
					case 7:
						{
						_localctx = new ConjExprContext(new ExprContext(_parentctx, _parentState));
						((ConjExprContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(341);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(342);
						((ConjExprContext)_localctx).op = match(T__27);
						setState(343);
						((ConjExprContext)_localctx).rhs = expr(7);
						}
						break;
					case 8:
						{
						_localctx = new DisjExprContext(new ExprContext(_parentctx, _parentState));
						((DisjExprContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(344);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(345);
						((DisjExprContext)_localctx).op = match(T__28);
						setState(346);
						((DisjExprContext)_localctx).rhs = expr(6);
						}
						break;
					case 9:
						{
						_localctx = new CondExprContext(new ExprContext(_parentctx, _parentState));
						((CondExprContext)_localctx).cond = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(347);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(348);
						match(T__29);
						setState(349);
						((CondExprContext)_localctx).thenExpr = expr(0);
						setState(350);
						match(T__8);
						setState(351);
						((CondExprContext)_localctx).elseExpr = expr(5);
						}
						break;
					case 10:
						{
						_localctx = new RightIdxFilterExprContext(new ExprContext(_parentctx, _parentState));
						((RightIdxFilterExprContext)_localctx).obj = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(353);
						if (!(precpred(_ctx, 14))) throw new FailedPredicateException(this, "precpred(_ctx, 14)");
						setState(354);
						match(T__14);
						setState(356);
						_errHandler.sync(this);
						_la = _input.LA(1);
						if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
							{
							setState(355);
							((RightIdxFilterExprContext)_localctx).rows = expr(0);
							}
						}

						setState(358);
						match(T__4);
						setState(360);
						_errHandler.sync(this);
						_la = _input.LA(1);
						if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
							{
							setState(359);
							((RightIdxFilterExprContext)_localctx).cols = expr(0);
							}
						}

						setState(362);
						match(T__15);
						}
						break;
					case 11:
						{
						_localctx = new RightIdxExtractExprContext(new ExprContext(_parentctx, _parentState));
						((RightIdxExtractExprContext)_localctx).obj = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(363);
						if (!(precpred(_ctx, 13))) throw new FailedPredicateException(this, "precpred(_ctx, 13)");
						setState(364);
						((RightIdxExtractExprContext)_localctx).idx = indexing();
						}
						break;
					}
					} 
				}
				setState(369);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,42,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FrameRowContext extends ParserRuleContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public FrameRowContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_frameRow; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterFrameRow(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitFrameRow(this);
		}
	}

	public final FrameRowContext frameRow() throws RecognitionException {
		FrameRowContext _localctx = new FrameRowContext(_ctx, getState());
		enterRule(_localctx, 32, RULE_frameRow);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(370);
			match(T__30);
			setState(379);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
				{
				setState(371);
				expr(0);
				setState(376);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__4) {
					{
					{
					setState(372);
					match(T__4);
					setState(373);
					expr(0);
					}
					}
					setState(378);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(381);
			match(T__31);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IndexingContext extends ParserRuleContext {
		public RangeContext rows;
		public RangeContext cols;
		public List<RangeContext> range() {
			return getRuleContexts(RangeContext.class);
		}
		public RangeContext range(int i) {
			return getRuleContext(RangeContext.class,i);
		}
		public IndexingContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_indexing; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterIndexing(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitIndexing(this);
		}
	}

	public final IndexingContext indexing() throws RecognitionException {
		IndexingContext _localctx = new IndexingContext(_ctx, getState());
		enterRule(_localctx, 34, RULE_indexing);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(383);
			match(T__30);
			setState(385);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519620L) != 0)) {
				{
				setState(384);
				((IndexingContext)_localctx).rows = range();
				}
			}

			setState(387);
			match(T__4);
			setState(389);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519620L) != 0)) {
				{
				setState(388);
				((IndexingContext)_localctx).cols = range();
				}
			}

			setState(391);
			match(T__31);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class RangeContext extends ParserRuleContext {
		public ExprContext pos;
		public ExprContext posLowerIncl;
		public ExprContext posUpperExcl;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public RangeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_range; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterRange(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitRange(this);
		}
	}

	public final RangeContext range() throws RecognitionException {
		RangeContext _localctx = new RangeContext(_ctx, getState());
		enterRule(_localctx, 36, RULE_range);
		int _la;
		try {
			setState(401);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,49,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(393);
				((RangeContext)_localctx).pos = expr(0);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				{
				setState(395);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
					{
					setState(394);
					((RangeContext)_localctx).posLowerIncl = expr(0);
					}
				}

				setState(397);
				match(T__8);
				setState(399);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2114912763519108L) != 0)) {
					{
					setState(398);
					((RangeContext)_localctx).posUpperExcl = expr(0);
					}
				}

				}
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class LiteralContext extends ParserRuleContext {
		public BoolLiteralContext bl;
		public TerminalNode INT_LITERAL() { return getToken(DaphneDSLGrammarParser.INT_LITERAL, 0); }
		public TerminalNode FLOAT_LITERAL() { return getToken(DaphneDSLGrammarParser.FLOAT_LITERAL, 0); }
		public BoolLiteralContext boolLiteral() {
			return getRuleContext(BoolLiteralContext.class,0);
		}
		public TerminalNode STRING_LITERAL() { return getToken(DaphneDSLGrammarParser.STRING_LITERAL, 0); }
		public LiteralContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_literal; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterLiteral(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitLiteral(this);
		}
	}

	public final LiteralContext literal() throws RecognitionException {
		LiteralContext _localctx = new LiteralContext(_ctx, getState());
		enterRule(_localctx, 38, RULE_literal);
		try {
			setState(407);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case INT_LITERAL:
				enterOuterAlt(_localctx, 1);
				{
				setState(403);
				match(INT_LITERAL);
				}
				break;
			case FLOAT_LITERAL:
				enterOuterAlt(_localctx, 2);
				{
				setState(404);
				match(FLOAT_LITERAL);
				}
				break;
			case KW_TRUE:
			case KW_FALSE:
				enterOuterAlt(_localctx, 3);
				{
				setState(405);
				((LiteralContext)_localctx).bl = boolLiteral();
				}
				break;
			case STRING_LITERAL:
				enterOuterAlt(_localctx, 4);
				{
				setState(406);
				match(STRING_LITERAL);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class BoolLiteralContext extends ParserRuleContext {
		public TerminalNode KW_TRUE() { return getToken(DaphneDSLGrammarParser.KW_TRUE, 0); }
		public TerminalNode KW_FALSE() { return getToken(DaphneDSLGrammarParser.KW_FALSE, 0); }
		public BoolLiteralContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_boolLiteral; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).enterBoolLiteral(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof DaphneDSLGrammarListener ) ((DaphneDSLGrammarListener)listener).exitBoolLiteral(this);
		}
	}

	public final BoolLiteralContext boolLiteral() throws RecognitionException {
		BoolLiteralContext _localctx = new BoolLiteralContext(_ctx, getState());
		enterRule(_localctx, 40, RULE_boolLiteral);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(409);
			_la = _input.LA(1);
			if ( !(_la==KW_TRUE || _la==KW_FALSE) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 15:
			return expr_sempred((ExprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 12);
		case 1:
			return precpred(_ctx, 11);
		case 2:
			return precpred(_ctx, 10);
		case 3:
			return precpred(_ctx, 9);
		case 4:
			return precpred(_ctx, 8);
		case 5:
			return precpred(_ctx, 7);
		case 6:
			return precpred(_ctx, 6);
		case 7:
			return precpred(_ctx, 5);
		case 8:
			return precpred(_ctx, 4);
		case 9:
			return precpred(_ctx, 14);
		case 10:
			return precpred(_ctx, 13);
		}
		return true;
	}

	public static final String _serializedATN =
		"\u0004\u00016\u019c\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001\u0002"+
		"\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004\u0007\u0004\u0002"+
		"\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007\u0007\u0007\u0002"+
		"\b\u0007\b\u0002\t\u0007\t\u0002\n\u0007\n\u0002\u000b\u0007\u000b\u0002"+
		"\f\u0007\f\u0002\r\u0007\r\u0002\u000e\u0007\u000e\u0002\u000f\u0007\u000f"+
		"\u0002\u0010\u0007\u0010\u0002\u0011\u0007\u0011\u0002\u0012\u0007\u0012"+
		"\u0002\u0013\u0007\u0013\u0002\u0014\u0007\u0014\u0001\u0000\u0005\u0000"+
		",\b\u0000\n\u0000\f\u0000/\t\u0000\u0001\u0000\u0001\u0000\u0001\u0001"+
		"\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"+
		"\u0001\u0001\u0001\u0001\u0003\u0001<\b\u0001\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0003\u0002B\b\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0003\u0001\u0003\u0005\u0003H\b\u0003\n\u0003\f\u0003K\t\u0003"+
		"\u0001\u0003\u0001\u0003\u0003\u0003O\b\u0003\u0001\u0004\u0001\u0004"+
		"\u0001\u0004\u0001\u0005\u0001\u0005\u0005\u0005V\b\u0005\n\u0005\f\u0005"+
		"Y\t\u0005\u0001\u0005\u0001\u0005\u0003\u0005]\b\u0005\u0001\u0005\u0001"+
		"\u0005\u0001\u0005\u0005\u0005b\b\u0005\n\u0005\f\u0005e\t\u0005\u0001"+
		"\u0005\u0001\u0005\u0003\u0005i\b\u0005\u0005\u0005k\b\u0005\n\u0005\f"+
		"\u0005n\t\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001"+
		"\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001"+
		"\u0006\u0003\u0006{\b\u0006\u0001\u0007\u0001\u0007\u0001\u0007\u0001"+
		"\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0001"+
		"\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0003\u0007\u008a\b\u0007\u0003"+
		"\u0007\u008c\b\u0007\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001"+
		"\b\u0001\b\u0001\b\u0003\b\u0097\b\b\u0001\b\u0001\b\u0001\b\u0001\t\u0001"+
		"\t\u0001\t\u0001\t\u0003\t\u00a0\b\t\u0001\t\u0001\t\u0001\t\u0003\t\u00a5"+
		"\b\t\u0001\t\u0001\t\u0001\n\u0001\n\u0001\n\u0001\n\u0005\n\u00ad\b\n"+
		"\n\n\f\n\u00b0\t\n\u0003\n\u00b2\b\n\u0001\n\u0001\n\u0001\u000b\u0001"+
		"\u000b\u0001\u000b\u0005\u000b\u00b9\b\u000b\n\u000b\f\u000b\u00bc\t\u000b"+
		"\u0001\u000b\u0003\u000b\u00bf\b\u000b\u0001\f\u0001\f\u0001\f\u0003\f"+
		"\u00c4\b\f\u0001\r\u0001\r\u0001\r\u0005\r\u00c9\b\r\n\r\f\r\u00cc\t\r"+
		"\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0003\u000e\u00d2\b\u000e"+
		"\u0001\u000e\u0003\u000e\u00d5\b\u000e\u0001\u000f\u0001\u000f\u0001\u000f"+
		"\u0001\u000f\u0001\u000f\u0001\u000f\u0005\u000f\u00dd\b\u000f\n\u000f"+
		"\f\u000f\u00e0\t\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f"+
		"\u0001\u000f\u0001\u000f\u0001\u000f\u0005\u000f\u00e9\b\u000f\n\u000f"+
		"\f\u000f\u00ec\t\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0003\u000f"+
		"\u00f1\b\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0005\u000f"+
		"\u00f7\b\u000f\n\u000f\f\u000f\u00fa\t\u000f\u0003\u000f\u00fc\b\u000f"+
		"\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f"+
		"\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0003\u000f"+
		"\u0109\b\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f"+
		"\u0001\u000f\u0001\u000f\u0001\u000f\u0005\u000f\u0113\b\u000f\n\u000f"+
		"\f\u000f\u0116\t\u000f\u0003\u000f\u0118\b\u000f\u0001\u000f\u0001\u000f"+
		"\u0001\u000f\u0003\u000f\u011d\b\u000f\u0001\u000f\u0001\u000f\u0003\u000f"+
		"\u0121\b\u000f\u0001\u000f\u0003\u000f\u0124\b\u000f\u0001\u000f\u0001"+
		"\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001"+
		"\u000f\u0001\u000f\u0005\u000f\u012f\b\u000f\n\u000f\f\u000f\u0132\t\u000f"+
		"\u0003\u000f\u0134\b\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f"+
		"\u0001\u000f\u0005\u000f\u013b\b\u000f\n\u000f\f\u000f\u013e\t\u000f\u0001"+
		"\u000f\u0001\u000f\u0003\u000f\u0142\b\u000f\u0001\u000f\u0001\u000f\u0001"+
		"\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001"+
		"\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001"+
		"\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001"+
		"\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001"+
		"\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001"+
		"\u000f\u0003\u000f\u0165\b\u000f\u0001\u000f\u0001\u000f\u0003\u000f\u0169"+
		"\b\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0005\u000f\u016e\b\u000f"+
		"\n\u000f\f\u000f\u0171\t\u000f\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0005\u0010\u0177\b\u0010\n\u0010\f\u0010\u017a\t\u0010\u0003\u0010"+
		"\u017c\b\u0010\u0001\u0010\u0001\u0010\u0001\u0011\u0001\u0011\u0003\u0011"+
		"\u0182\b\u0011\u0001\u0011\u0001\u0011\u0003\u0011\u0186\b\u0011\u0001"+
		"\u0011\u0001\u0011\u0001\u0012\u0001\u0012\u0003\u0012\u018c\b\u0012\u0001"+
		"\u0012\u0001\u0012\u0003\u0012\u0190\b\u0012\u0003\u0012\u0192\b\u0012"+
		"\u0001\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0003\u0013\u0198\b\u0013"+
		"\u0001\u0014\u0001\u0014\u0001\u0014\u0000\u0001\u001e\u0015\u0000\u0002"+
		"\u0004\u0006\b\n\f\u000e\u0010\u0012\u0014\u0016\u0018\u001a\u001c\u001e"+
		" \"$&(\u0000\u0004\u0001\u0000\u0014\u0015\u0001\u0000\u0016\u0017\u0002"+
		"\u0000\u000b\f\u0018\u001b\u0001\u0000\'(\u01d3\u0000-\u0001\u0000\u0000"+
		"\u0000\u0002;\u0001\u0000\u0000\u0000\u0004=\u0001\u0000\u0000\u0000\u0006"+
		"E\u0001\u0000\u0000\u0000\bP\u0001\u0000\u0000\u0000\nW\u0001\u0000\u0000"+
		"\u0000\fs\u0001\u0000\u0000\u0000\u000e\u008b\u0001\u0000\u0000\u0000"+
		"\u0010\u008d\u0001\u0000\u0000\u0000\u0012\u009b\u0001\u0000\u0000\u0000"+
		"\u0014\u00a8\u0001\u0000\u0000\u0000\u0016\u00b5\u0001\u0000\u0000\u0000"+
		"\u0018\u00c0\u0001\u0000\u0000\u0000\u001a\u00c5\u0001\u0000\u0000\u0000"+
		"\u001c\u00d4\u0001\u0000\u0000\u0000\u001e\u0141\u0001\u0000\u0000\u0000"+
		" \u0172\u0001\u0000\u0000\u0000\"\u017f\u0001\u0000\u0000\u0000$\u0191"+
		"\u0001\u0000\u0000\u0000&\u0197\u0001\u0000\u0000\u0000(\u0199\u0001\u0000"+
		"\u0000\u0000*,\u0003\u0002\u0001\u0000+*\u0001\u0000\u0000\u0000,/\u0001"+
		"\u0000\u0000\u0000-+\u0001\u0000\u0000\u0000-.\u0001\u0000\u0000\u0000"+
		".0\u0001\u0000\u0000\u0000/-\u0001\u0000\u0000\u000001\u0005\u0000\u0000"+
		"\u00011\u0001\u0001\u0000\u0000\u00002<\u0003\u0006\u0003\u00003<\u0003"+
		"\b\u0004\u00004<\u0003\n\u0005\u00005<\u0003\f\u0006\u00006<\u0003\u000e"+
		"\u0007\u00007<\u0003\u0010\b\u00008<\u0003\u0012\t\u00009<\u0003\u0014"+
		"\n\u0000:<\u0003\u0004\u0002\u0000;2\u0001\u0000\u0000\u0000;3\u0001\u0000"+
		"\u0000\u0000;4\u0001\u0000\u0000\u0000;5\u0001\u0000\u0000\u0000;6\u0001"+
		"\u0000\u0000\u0000;7\u0001\u0000\u0000\u0000;8\u0001\u0000\u0000\u0000"+
		";9\u0001\u0000\u0000\u0000;:\u0001\u0000\u0000\u0000<\u0003\u0001\u0000"+
		"\u0000\u0000=>\u0005,\u0000\u0000>A\u00051\u0000\u0000?@\u0005)\u0000"+
		"\u0000@B\u00051\u0000\u0000A?\u0001\u0000\u0000\u0000AB\u0001\u0000\u0000"+
		"\u0000BC\u0001\u0000\u0000\u0000CD\u0005\u0001\u0000\u0000D\u0005\u0001"+
		"\u0000\u0000\u0000EI\u0005\u0002\u0000\u0000FH\u0003\u0002\u0001\u0000"+
		"GF\u0001\u0000\u0000\u0000HK\u0001\u0000\u0000\u0000IG\u0001\u0000\u0000"+
		"\u0000IJ\u0001\u0000\u0000\u0000JL\u0001\u0000\u0000\u0000KI\u0001\u0000"+
		"\u0000\u0000LN\u0005\u0003\u0000\u0000MO\u0005\u0001\u0000\u0000NM\u0001"+
		"\u0000\u0000\u0000NO\u0001\u0000\u0000\u0000O\u0007\u0001\u0000\u0000"+
		"\u0000PQ\u0003\u001e\u000f\u0000QR\u0005\u0001\u0000\u0000R\t\u0001\u0000"+
		"\u0000\u0000ST\u00052\u0000\u0000TV\u0005\u0004\u0000\u0000US\u0001\u0000"+
		"\u0000\u0000VY\u0001\u0000\u0000\u0000WU\u0001\u0000\u0000\u0000WX\u0001"+
		"\u0000\u0000\u0000XZ\u0001\u0000\u0000\u0000YW\u0001\u0000\u0000\u0000"+
		"Z\\\u00052\u0000\u0000[]\u0003\"\u0011\u0000\\[\u0001\u0000\u0000\u0000"+
		"\\]\u0001\u0000\u0000\u0000]l\u0001\u0000\u0000\u0000^c\u0005\u0005\u0000"+
		"\u0000_`\u00052\u0000\u0000`b\u0005\u0004\u0000\u0000a_\u0001\u0000\u0000"+
		"\u0000be\u0001\u0000\u0000\u0000ca\u0001\u0000\u0000\u0000cd\u0001\u0000"+
		"\u0000\u0000df\u0001\u0000\u0000\u0000ec\u0001\u0000\u0000\u0000fh\u0005"+
		"2\u0000\u0000gi\u0003\"\u0011\u0000hg\u0001\u0000\u0000\u0000hi\u0001"+
		"\u0000\u0000\u0000ik\u0001\u0000\u0000\u0000j^\u0001\u0000\u0000\u0000"+
		"kn\u0001\u0000\u0000\u0000lj\u0001\u0000\u0000\u0000lm\u0001\u0000\u0000"+
		"\u0000mo\u0001\u0000\u0000\u0000nl\u0001\u0000\u0000\u0000op\u0005\u0006"+
		"\u0000\u0000pq\u0003\u001e\u000f\u0000qr\u0005\u0001\u0000\u0000r\u000b"+
		"\u0001\u0000\u0000\u0000st\u0005!\u0000\u0000tu\u0005\u0007\u0000\u0000"+
		"uv\u0003\u001e\u000f\u0000vw\u0005\b\u0000\u0000wz\u0003\u0002\u0001\u0000"+
		"xy\u0005\"\u0000\u0000y{\u0003\u0002\u0001\u0000zx\u0001\u0000\u0000\u0000"+
		"z{\u0001\u0000\u0000\u0000{\r\u0001\u0000\u0000\u0000|}\u0005#\u0000\u0000"+
		"}~\u0005\u0007\u0000\u0000~\u007f\u0003\u001e\u000f\u0000\u007f\u0080"+
		"\u0005\b\u0000\u0000\u0080\u0081\u0003\u0002\u0001\u0000\u0081\u008c\u0001"+
		"\u0000\u0000\u0000\u0082\u0083\u0005$\u0000\u0000\u0083\u0084\u0003\u0002"+
		"\u0001\u0000\u0084\u0085\u0005#\u0000\u0000\u0085\u0086\u0005\u0007\u0000"+
		"\u0000\u0086\u0087\u0003\u001e\u000f\u0000\u0087\u0089\u0005\b\u0000\u0000"+
		"\u0088\u008a\u0005\u0001\u0000\u0000\u0089\u0088\u0001\u0000\u0000\u0000"+
		"\u0089\u008a\u0001\u0000\u0000\u0000\u008a\u008c\u0001\u0000\u0000\u0000"+
		"\u008b|\u0001\u0000\u0000\u0000\u008b\u0082\u0001\u0000\u0000\u0000\u008c"+
		"\u000f\u0001\u0000\u0000\u0000\u008d\u008e\u0005%\u0000\u0000\u008e\u008f"+
		"\u0005\u0007\u0000\u0000\u008f\u0090\u00052\u0000\u0000\u0090\u0091\u0005"+
		"&\u0000\u0000\u0091\u0092\u0003\u001e\u000f\u0000\u0092\u0093\u0005\t"+
		"\u0000\u0000\u0093\u0096\u0003\u001e\u000f\u0000\u0094\u0095\u0005\t\u0000"+
		"\u0000\u0095\u0097\u0003\u001e\u000f\u0000\u0096\u0094\u0001\u0000\u0000"+
		"\u0000\u0096\u0097\u0001\u0000\u0000\u0000\u0097\u0098\u0001\u0000\u0000"+
		"\u0000\u0098\u0099\u0005\b\u0000\u0000\u0099\u009a\u0003\u0002\u0001\u0000"+
		"\u009a\u0011\u0001\u0000\u0000\u0000\u009b\u009c\u0005*\u0000\u0000\u009c"+
		"\u009d\u00052\u0000\u0000\u009d\u009f\u0005\u0007\u0000\u0000\u009e\u00a0"+
		"\u0003\u0016\u000b\u0000\u009f\u009e\u0001\u0000\u0000\u0000\u009f\u00a0"+
		"\u0001\u0000\u0000\u0000\u00a0\u00a1\u0001\u0000\u0000\u0000\u00a1\u00a4"+
		"\u0005\b\u0000\u0000\u00a2\u00a3\u0005\n\u0000\u0000\u00a3\u00a5\u0003"+
		"\u001a\r\u0000\u00a4\u00a2\u0001\u0000\u0000\u0000\u00a4\u00a5\u0001\u0000"+
		"\u0000\u0000\u00a5\u00a6\u0001\u0000\u0000\u0000\u00a6\u00a7\u0003\u0006"+
		"\u0003\u0000\u00a7\u0013\u0001\u0000\u0000\u0000\u00a8\u00b1\u0005+\u0000"+
		"\u0000\u00a9\u00ae\u0003\u001e\u000f\u0000\u00aa\u00ab\u0005\u0005\u0000"+
		"\u0000\u00ab\u00ad\u0003\u001e\u000f\u0000\u00ac\u00aa\u0001\u0000\u0000"+
		"\u0000\u00ad\u00b0\u0001\u0000\u0000\u0000\u00ae\u00ac\u0001\u0000\u0000"+
		"\u0000\u00ae\u00af\u0001\u0000\u0000\u0000\u00af\u00b2\u0001\u0000\u0000"+
		"\u0000\u00b0\u00ae\u0001\u0000\u0000\u0000\u00b1\u00a9\u0001\u0000\u0000"+
		"\u0000\u00b1\u00b2\u0001\u0000\u0000\u0000\u00b2\u00b3\u0001\u0000\u0000"+
		"\u0000\u00b3\u00b4\u0005\u0001\u0000\u0000\u00b4\u0015\u0001\u0000\u0000"+
		"\u0000\u00b5\u00ba\u0003\u0018\f\u0000\u00b6\u00b7\u0005\u0005\u0000\u0000"+
		"\u00b7\u00b9\u0003\u0018\f\u0000\u00b8\u00b6\u0001\u0000\u0000\u0000\u00b9"+
		"\u00bc\u0001\u0000\u0000\u0000\u00ba\u00b8\u0001\u0000\u0000\u0000\u00ba"+
		"\u00bb\u0001\u0000\u0000\u0000\u00bb\u00be\u0001\u0000\u0000\u0000\u00bc"+
		"\u00ba\u0001\u0000\u0000\u0000\u00bd\u00bf\u0005\u0005\u0000\u0000\u00be"+
		"\u00bd\u0001\u0000\u0000\u0000\u00be\u00bf\u0001\u0000\u0000\u0000\u00bf"+
		"\u0017\u0001\u0000\u0000\u0000\u00c0\u00c3\u00052\u0000\u0000\u00c1\u00c2"+
		"\u0005\t\u0000\u0000\u00c2\u00c4\u0003\u001c\u000e\u0000\u00c3\u00c1\u0001"+
		"\u0000\u0000\u0000\u00c3\u00c4\u0001\u0000\u0000\u0000\u00c4\u0019\u0001"+
		"\u0000\u0000\u0000\u00c5\u00ca\u0003\u001c\u000e\u0000\u00c6\u00c7\u0005"+
		"\u0005\u0000\u0000\u00c7\u00c9\u0003\u001c\u000e\u0000\u00c8\u00c6\u0001"+
		"\u0000\u0000\u0000\u00c9\u00cc\u0001\u0000\u0000\u0000\u00ca\u00c8\u0001"+
		"\u0000\u0000\u0000\u00ca\u00cb\u0001\u0000\u0000\u0000\u00cb\u001b\u0001"+
		"\u0000\u0000\u0000\u00cc\u00ca\u0001\u0000\u0000\u0000\u00cd\u00d1\u0005"+
		"-\u0000\u0000\u00ce\u00cf\u0005\u000b\u0000\u0000\u00cf\u00d0\u0005.\u0000"+
		"\u0000\u00d0\u00d2\u0005\f\u0000\u0000\u00d1\u00ce\u0001\u0000\u0000\u0000"+
		"\u00d1\u00d2\u0001\u0000\u0000\u0000\u00d2\u00d5\u0001\u0000\u0000\u0000"+
		"\u00d3\u00d5\u0005.\u0000\u0000\u00d4\u00cd\u0001\u0000\u0000\u0000\u00d4"+
		"\u00d3\u0001\u0000\u0000\u0000\u00d5\u001d\u0001\u0000\u0000\u0000\u00d6"+
		"\u00d7\u0006\u000f\uffff\uffff\u0000\u00d7\u0142\u0003&\u0013\u0000\u00d8"+
		"\u00d9\u0005\r\u0000\u0000\u00d9\u0142\u00052\u0000\u0000\u00da\u00db"+
		"\u00052\u0000\u0000\u00db\u00dd\u0005\u0004\u0000\u0000\u00dc\u00da\u0001"+
		"\u0000\u0000\u0000\u00dd\u00e0\u0001\u0000\u0000\u0000\u00de\u00dc\u0001"+
		"\u0000\u0000\u0000\u00de\u00df\u0001\u0000\u0000\u0000\u00df\u00e1\u0001"+
		"\u0000\u0000\u0000\u00e0\u00de\u0001\u0000\u0000\u0000\u00e1\u0142\u0005"+
		"2\u0000\u0000\u00e2\u00e3\u0005\u0007\u0000\u0000\u00e3\u00e4\u0003\u001e"+
		"\u000f\u0000\u00e4\u00e5\u0005\b\u0000\u0000\u00e5\u0142\u0001\u0000\u0000"+
		"\u0000\u00e6\u00e7\u00052\u0000\u0000\u00e7\u00e9\u0005\u0004\u0000\u0000"+
		"\u00e8\u00e6\u0001\u0000\u0000\u0000\u00e9\u00ec\u0001\u0000\u0000\u0000"+
		"\u00ea\u00e8\u0001\u0000\u0000\u0000\u00ea\u00eb\u0001\u0000\u0000\u0000"+
		"\u00eb\u00ed\u0001\u0000\u0000\u0000\u00ec\u00ea\u0001\u0000\u0000\u0000"+
		"\u00ed\u00f0\u00052\u0000\u0000\u00ee\u00ef\u0005\u000e\u0000\u0000\u00ef"+
		"\u00f1\u00052\u0000\u0000\u00f0\u00ee\u0001\u0000\u0000\u0000\u00f0\u00f1"+
		"\u0001\u0000\u0000\u0000\u00f1\u00f2\u0001\u0000\u0000\u0000\u00f2\u00fb"+
		"\u0005\u0007\u0000\u0000\u00f3\u00f8\u0003\u001e\u000f\u0000\u00f4\u00f5"+
		"\u0005\u0005\u0000\u0000\u00f5\u00f7\u0003\u001e\u000f\u0000\u00f6\u00f4"+
		"\u0001\u0000\u0000\u0000\u00f7\u00fa\u0001\u0000\u0000\u0000\u00f8\u00f6"+
		"\u0001\u0000\u0000\u0000\u00f8\u00f9\u0001\u0000\u0000\u0000\u00f9\u00fc"+
		"\u0001\u0000\u0000\u0000\u00fa\u00f8\u0001\u0000\u0000\u0000\u00fb\u00f3"+
		"\u0001\u0000\u0000\u0000\u00fb\u00fc\u0001\u0000\u0000\u0000\u00fc\u00fd"+
		"\u0001\u0000\u0000\u0000\u00fd\u0142\u0005\b\u0000\u0000\u00fe\u0108\u0005"+
		")\u0000\u0000\u00ff\u0100\u0005\u0004\u0000\u0000\u0100\u0109\u0005-\u0000"+
		"\u0000\u0101\u0102\u0005\u0004\u0000\u0000\u0102\u0109\u0005.\u0000\u0000"+
		"\u0103\u0104\u0005\u0004\u0000\u0000\u0104\u0105\u0005-\u0000\u0000\u0105"+
		"\u0106\u0005\u000b\u0000\u0000\u0106\u0107\u0005.\u0000\u0000\u0107\u0109"+
		"\u0005\f\u0000\u0000\u0108\u00ff\u0001\u0000\u0000\u0000\u0108\u0101\u0001"+
		"\u0000\u0000\u0000\u0108\u0103\u0001\u0000\u0000\u0000\u0109\u010a\u0001"+
		"\u0000\u0000\u0000\u010a\u010b\u0005\u0007\u0000\u0000\u010b\u010c\u0003"+
		"\u001e\u000f\u0000\u010c\u010d\u0005\b\u0000\u0000\u010d\u0142\u0001\u0000"+
		"\u0000\u0000\u010e\u0117\u0005\u001f\u0000\u0000\u010f\u0114\u0003\u001e"+
		"\u000f\u0000\u0110\u0111\u0005\u0005\u0000\u0000\u0111\u0113\u0003\u001e"+
		"\u000f\u0000\u0112\u0110\u0001\u0000\u0000\u0000\u0113\u0116\u0001\u0000"+
		"\u0000\u0000\u0114\u0112\u0001\u0000\u0000\u0000\u0114\u0115\u0001\u0000"+
		"\u0000\u0000\u0115\u0118\u0001\u0000\u0000\u0000\u0116\u0114\u0001\u0000"+
		"\u0000\u0000\u0117\u010f\u0001\u0000\u0000\u0000\u0117\u0118\u0001\u0000"+
		"\u0000\u0000\u0118\u0119\u0001\u0000\u0000\u0000\u0119\u0123\u0005 \u0000"+
		"\u0000\u011a\u011c\u0005\u0007\u0000\u0000\u011b\u011d\u0003\u001e\u000f"+
		"\u0000\u011c\u011b\u0001\u0000\u0000\u0000\u011c\u011d\u0001\u0000\u0000"+
		"\u0000\u011d\u011e\u0001\u0000\u0000\u0000\u011e\u0120\u0005\u0005\u0000"+
		"\u0000\u011f\u0121\u0003\u001e\u000f\u0000\u0120\u011f\u0001\u0000\u0000"+
		"\u0000\u0120\u0121\u0001\u0000\u0000\u0000\u0121\u0122\u0001\u0000\u0000"+
		"\u0000\u0122\u0124\u0005\b\u0000\u0000\u0123\u011a\u0001\u0000\u0000\u0000"+
		"\u0123\u0124\u0001\u0000\u0000\u0000\u0124\u0142\u0001\u0000\u0000\u0000"+
		"\u0125\u0133\u0005\u0002\u0000\u0000\u0126\u0127\u0003\u001e\u000f\u0000"+
		"\u0127\u0128\u0005\t\u0000\u0000\u0128\u0130\u0003\u001e\u000f\u0000\u0129"+
		"\u012a\u0005\u0005\u0000\u0000\u012a\u012b\u0003\u001e\u000f\u0000\u012b"+
		"\u012c\u0005\t\u0000\u0000\u012c\u012d\u0003\u001e\u000f\u0000\u012d\u012f"+
		"\u0001\u0000\u0000\u0000\u012e\u0129\u0001\u0000\u0000\u0000\u012f\u0132"+
		"\u0001\u0000\u0000\u0000\u0130\u012e\u0001\u0000\u0000\u0000\u0130\u0131"+
		"\u0001\u0000\u0000\u0000\u0131\u0134\u0001\u0000\u0000\u0000\u0132\u0130"+
		"\u0001\u0000\u0000\u0000\u0133\u0126\u0001\u0000\u0000\u0000\u0133\u0134"+
		"\u0001\u0000\u0000\u0000\u0134\u0135\u0001\u0000\u0000\u0000\u0135\u0142"+
		"\u0005\u0003\u0000\u0000\u0136\u0137\u0005\u0002\u0000\u0000\u0137\u013c"+
		"\u0003 \u0010\u0000\u0138\u0139\u0005\u0005\u0000\u0000\u0139\u013b\u0003"+
		" \u0010\u0000\u013a\u0138\u0001\u0000\u0000\u0000\u013b\u013e\u0001\u0000"+
		"\u0000\u0000\u013c\u013a\u0001\u0000\u0000\u0000\u013c\u013d\u0001\u0000"+
		"\u0000\u0000\u013d\u013f\u0001\u0000\u0000\u0000\u013e\u013c\u0001\u0000"+
		"\u0000\u0000\u013f\u0140\u0005\u0003\u0000\u0000\u0140\u0142\u0001\u0000"+
		"\u0000\u0000\u0141\u00d6\u0001\u0000\u0000\u0000\u0141\u00d8\u0001\u0000"+
		"\u0000\u0000\u0141\u00de\u0001\u0000\u0000\u0000\u0141\u00e2\u0001\u0000"+
		"\u0000\u0000\u0141\u00ea\u0001\u0000\u0000\u0000\u0141\u00fe\u0001\u0000"+
		"\u0000\u0000\u0141\u010e\u0001\u0000\u0000\u0000\u0141\u0125\u0001\u0000"+
		"\u0000\u0000\u0141\u0136\u0001\u0000\u0000\u0000\u0142\u016f\u0001\u0000"+
		"\u0000\u0000\u0143\u0144\n\f\u0000\u0000\u0144\u0145\u0005\u0011\u0000"+
		"\u0000\u0145\u016e\u0003\u001e\u000f\r\u0146\u0147\n\u000b\u0000\u0000"+
		"\u0147\u0148\u0005\u0012\u0000\u0000\u0148\u016e\u0003\u001e\u000f\f\u0149"+
		"\u014a\n\n\u0000\u0000\u014a\u014b\u0005\u0013\u0000\u0000\u014b\u016e"+
		"\u0003\u001e\u000f\u000b\u014c\u014d\n\t\u0000\u0000\u014d\u014e\u0007"+
		"\u0000\u0000\u0000\u014e\u016e\u0003\u001e\u000f\n\u014f\u0150\n\b\u0000"+
		"\u0000\u0150\u0151\u0007\u0001\u0000\u0000\u0151\u016e\u0003\u001e\u000f"+
		"\t\u0152\u0153\n\u0007\u0000\u0000\u0153\u0154\u0007\u0002\u0000\u0000"+
		"\u0154\u016e\u0003\u001e\u000f\b\u0155\u0156\n\u0006\u0000\u0000\u0156"+
		"\u0157\u0005\u001c\u0000\u0000\u0157\u016e\u0003\u001e\u000f\u0007\u0158"+
		"\u0159\n\u0005\u0000\u0000\u0159\u015a\u0005\u001d\u0000\u0000\u015a\u016e"+
		"\u0003\u001e\u000f\u0006\u015b\u015c\n\u0004\u0000\u0000\u015c\u015d\u0005"+
		"\u001e\u0000\u0000\u015d\u015e\u0003\u001e\u000f\u0000\u015e\u015f\u0005"+
		"\t\u0000\u0000\u015f\u0160\u0003\u001e\u000f\u0005\u0160\u016e\u0001\u0000"+
		"\u0000\u0000\u0161\u0162\n\u000e\u0000\u0000\u0162\u0164\u0005\u000f\u0000"+
		"\u0000\u0163\u0165\u0003\u001e\u000f\u0000\u0164\u0163\u0001\u0000\u0000"+
		"\u0000\u0164\u0165\u0001\u0000\u0000\u0000\u0165\u0166\u0001\u0000\u0000"+
		"\u0000\u0166\u0168\u0005\u0005\u0000\u0000\u0167\u0169\u0003\u001e\u000f"+
		"\u0000\u0168\u0167\u0001\u0000\u0000\u0000\u0168\u0169\u0001\u0000\u0000"+
		"\u0000\u0169\u016a\u0001\u0000\u0000\u0000\u016a\u016e\u0005\u0010\u0000"+
		"\u0000\u016b\u016c\n\r\u0000\u0000\u016c\u016e\u0003\"\u0011\u0000\u016d"+
		"\u0143\u0001\u0000\u0000\u0000\u016d\u0146\u0001\u0000\u0000\u0000\u016d"+
		"\u0149\u0001\u0000\u0000\u0000\u016d\u014c\u0001\u0000\u0000\u0000\u016d"+
		"\u014f\u0001\u0000\u0000\u0000\u016d\u0152\u0001\u0000\u0000\u0000\u016d"+
		"\u0155\u0001\u0000\u0000\u0000\u016d\u0158\u0001\u0000\u0000\u0000\u016d"+
		"\u015b\u0001\u0000\u0000\u0000\u016d\u0161\u0001\u0000\u0000\u0000\u016d"+
		"\u016b\u0001\u0000\u0000\u0000\u016e\u0171\u0001\u0000\u0000\u0000\u016f"+
		"\u016d\u0001\u0000\u0000\u0000\u016f\u0170\u0001\u0000\u0000\u0000\u0170"+
		"\u001f\u0001\u0000\u0000\u0000\u0171\u016f\u0001\u0000\u0000\u0000\u0172"+
		"\u017b\u0005\u001f\u0000\u0000\u0173\u0178\u0003\u001e\u000f\u0000\u0174"+
		"\u0175\u0005\u0005\u0000\u0000\u0175\u0177\u0003\u001e\u000f\u0000\u0176"+
		"\u0174\u0001\u0000\u0000\u0000\u0177\u017a\u0001\u0000\u0000\u0000\u0178"+
		"\u0176\u0001\u0000\u0000\u0000\u0178\u0179\u0001\u0000\u0000\u0000\u0179"+
		"\u017c\u0001\u0000\u0000\u0000\u017a\u0178\u0001\u0000\u0000\u0000\u017b"+
		"\u0173\u0001\u0000\u0000\u0000\u017b\u017c\u0001\u0000\u0000\u0000\u017c"+
		"\u017d\u0001\u0000\u0000\u0000\u017d\u017e\u0005 \u0000\u0000\u017e!\u0001"+
		"\u0000\u0000\u0000\u017f\u0181\u0005\u001f\u0000\u0000\u0180\u0182\u0003"+
		"$\u0012\u0000\u0181\u0180\u0001\u0000\u0000\u0000\u0181\u0182\u0001\u0000"+
		"\u0000\u0000\u0182\u0183\u0001\u0000\u0000\u0000\u0183\u0185\u0005\u0005"+
		"\u0000\u0000\u0184\u0186\u0003$\u0012\u0000\u0185\u0184\u0001\u0000\u0000"+
		"\u0000\u0185\u0186\u0001\u0000\u0000\u0000\u0186\u0187\u0001\u0000\u0000"+
		"\u0000\u0187\u0188\u0005 \u0000\u0000\u0188#\u0001\u0000\u0000\u0000\u0189"+
		"\u0192\u0003\u001e\u000f\u0000\u018a\u018c\u0003\u001e\u000f\u0000\u018b"+
		"\u018a\u0001\u0000\u0000\u0000\u018b\u018c\u0001\u0000\u0000\u0000\u018c"+
		"\u018d\u0001\u0000\u0000\u0000\u018d\u018f\u0005\t\u0000\u0000\u018e\u0190"+
		"\u0003\u001e\u000f\u0000\u018f\u018e\u0001\u0000\u0000\u0000\u018f\u0190"+
		"\u0001\u0000\u0000\u0000\u0190\u0192\u0001\u0000\u0000\u0000\u0191\u0189"+
		"\u0001\u0000\u0000\u0000\u0191\u018b\u0001\u0000\u0000\u0000\u0192%\u0001"+
		"\u0000\u0000\u0000\u0193\u0198\u0005/\u0000\u0000\u0194\u0198\u00050\u0000"+
		"\u0000\u0195\u0198\u0003(\u0014\u0000\u0196\u0198\u00051\u0000\u0000\u0197"+
		"\u0193\u0001\u0000\u0000\u0000\u0197\u0194\u0001\u0000\u0000\u0000\u0197"+
		"\u0195\u0001\u0000\u0000\u0000\u0197\u0196\u0001\u0000\u0000\u0000\u0198"+
		"\'\u0001\u0000\u0000\u0000\u0199\u019a\u0007\u0003\u0000\u0000\u019a)"+
		"\u0001\u0000\u0000\u00003-;AINW\\chlz\u0089\u008b\u0096\u009f\u00a4\u00ae"+
		"\u00b1\u00ba\u00be\u00c3\u00ca\u00d1\u00d4\u00de\u00ea\u00f0\u00f8\u00fb"+
		"\u0108\u0114\u0117\u011c\u0120\u0123\u0130\u0133\u013c\u0141\u0164\u0168"+
		"\u016d\u016f\u0178\u017b\u0181\u0185\u018b\u018f\u0191\u0197";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}