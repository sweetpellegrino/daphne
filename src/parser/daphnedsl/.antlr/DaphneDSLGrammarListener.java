// Generated from /home/niklas/daphne/src/parser/daphnedsl/DaphneDSLGrammar.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link DaphneDSLGrammarParser}.
 */
public interface DaphneDSLGrammarListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#script}.
	 * @param ctx the parse tree
	 */
	void enterScript(DaphneDSLGrammarParser.ScriptContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#script}.
	 * @param ctx the parse tree
	 */
	void exitScript(DaphneDSLGrammarParser.ScriptContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterStatement(DaphneDSLGrammarParser.StatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitStatement(DaphneDSLGrammarParser.StatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#importStatement}.
	 * @param ctx the parse tree
	 */
	void enterImportStatement(DaphneDSLGrammarParser.ImportStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#importStatement}.
	 * @param ctx the parse tree
	 */
	void exitImportStatement(DaphneDSLGrammarParser.ImportStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#blockStatement}.
	 * @param ctx the parse tree
	 */
	void enterBlockStatement(DaphneDSLGrammarParser.BlockStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#blockStatement}.
	 * @param ctx the parse tree
	 */
	void exitBlockStatement(DaphneDSLGrammarParser.BlockStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#exprStatement}.
	 * @param ctx the parse tree
	 */
	void enterExprStatement(DaphneDSLGrammarParser.ExprStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#exprStatement}.
	 * @param ctx the parse tree
	 */
	void exitExprStatement(DaphneDSLGrammarParser.ExprStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#assignStatement}.
	 * @param ctx the parse tree
	 */
	void enterAssignStatement(DaphneDSLGrammarParser.AssignStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#assignStatement}.
	 * @param ctx the parse tree
	 */
	void exitAssignStatement(DaphneDSLGrammarParser.AssignStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#ifStatement}.
	 * @param ctx the parse tree
	 */
	void enterIfStatement(DaphneDSLGrammarParser.IfStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#ifStatement}.
	 * @param ctx the parse tree
	 */
	void exitIfStatement(DaphneDSLGrammarParser.IfStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#whileStatement}.
	 * @param ctx the parse tree
	 */
	void enterWhileStatement(DaphneDSLGrammarParser.WhileStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#whileStatement}.
	 * @param ctx the parse tree
	 */
	void exitWhileStatement(DaphneDSLGrammarParser.WhileStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#forStatement}.
	 * @param ctx the parse tree
	 */
	void enterForStatement(DaphneDSLGrammarParser.ForStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#forStatement}.
	 * @param ctx the parse tree
	 */
	void exitForStatement(DaphneDSLGrammarParser.ForStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void enterFunctionStatement(DaphneDSLGrammarParser.FunctionStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void exitFunctionStatement(DaphneDSLGrammarParser.FunctionStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#returnStatement}.
	 * @param ctx the parse tree
	 */
	void enterReturnStatement(DaphneDSLGrammarParser.ReturnStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#returnStatement}.
	 * @param ctx the parse tree
	 */
	void exitReturnStatement(DaphneDSLGrammarParser.ReturnStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#functionArgs}.
	 * @param ctx the parse tree
	 */
	void enterFunctionArgs(DaphneDSLGrammarParser.FunctionArgsContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#functionArgs}.
	 * @param ctx the parse tree
	 */
	void exitFunctionArgs(DaphneDSLGrammarParser.FunctionArgsContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#functionArg}.
	 * @param ctx the parse tree
	 */
	void enterFunctionArg(DaphneDSLGrammarParser.FunctionArgContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#functionArg}.
	 * @param ctx the parse tree
	 */
	void exitFunctionArg(DaphneDSLGrammarParser.FunctionArgContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#functionRetTypes}.
	 * @param ctx the parse tree
	 */
	void enterFunctionRetTypes(DaphneDSLGrammarParser.FunctionRetTypesContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#functionRetTypes}.
	 * @param ctx the parse tree
	 */
	void exitFunctionRetTypes(DaphneDSLGrammarParser.FunctionRetTypesContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#funcTypeDef}.
	 * @param ctx the parse tree
	 */
	void enterFuncTypeDef(DaphneDSLGrammarParser.FuncTypeDefContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#funcTypeDef}.
	 * @param ctx the parse tree
	 */
	void exitFuncTypeDef(DaphneDSLGrammarParser.FuncTypeDefContext ctx);
	/**
	 * Enter a parse tree produced by the {@code rightIdxExtractExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterRightIdxExtractExpr(DaphneDSLGrammarParser.RightIdxExtractExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code rightIdxExtractExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitRightIdxExtractExpr(DaphneDSLGrammarParser.RightIdxExtractExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code modExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterModExpr(DaphneDSLGrammarParser.ModExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code modExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitModExpr(DaphneDSLGrammarParser.ModExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code castExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterCastExpr(DaphneDSLGrammarParser.CastExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code castExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitCastExpr(DaphneDSLGrammarParser.CastExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code matmulExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterMatmulExpr(DaphneDSLGrammarParser.MatmulExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code matmulExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitMatmulExpr(DaphneDSLGrammarParser.MatmulExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code condExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterCondExpr(DaphneDSLGrammarParser.CondExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code condExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitCondExpr(DaphneDSLGrammarParser.CondExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code conjExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterConjExpr(DaphneDSLGrammarParser.ConjExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code conjExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitConjExpr(DaphneDSLGrammarParser.ConjExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code disjExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterDisjExpr(DaphneDSLGrammarParser.DisjExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code disjExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitDisjExpr(DaphneDSLGrammarParser.DisjExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code rightIdxFilterExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterRightIdxFilterExpr(DaphneDSLGrammarParser.RightIdxFilterExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code rightIdxFilterExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitRightIdxFilterExpr(DaphneDSLGrammarParser.RightIdxFilterExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code matrixLiteralExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterMatrixLiteralExpr(DaphneDSLGrammarParser.MatrixLiteralExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code matrixLiteralExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitMatrixLiteralExpr(DaphneDSLGrammarParser.MatrixLiteralExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code paranthesesExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterParanthesesExpr(DaphneDSLGrammarParser.ParanthesesExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code paranthesesExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitParanthesesExpr(DaphneDSLGrammarParser.ParanthesesExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code cmpExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterCmpExpr(DaphneDSLGrammarParser.CmpExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code cmpExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitCmpExpr(DaphneDSLGrammarParser.CmpExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code addExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterAddExpr(DaphneDSLGrammarParser.AddExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code addExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitAddExpr(DaphneDSLGrammarParser.AddExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code literalExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterLiteralExpr(DaphneDSLGrammarParser.LiteralExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code literalExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitLiteralExpr(DaphneDSLGrammarParser.LiteralExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code rowMajorFrameLiteralExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterRowMajorFrameLiteralExpr(DaphneDSLGrammarParser.RowMajorFrameLiteralExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code rowMajorFrameLiteralExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitRowMajorFrameLiteralExpr(DaphneDSLGrammarParser.RowMajorFrameLiteralExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code mulExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterMulExpr(DaphneDSLGrammarParser.MulExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code mulExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitMulExpr(DaphneDSLGrammarParser.MulExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code argExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterArgExpr(DaphneDSLGrammarParser.ArgExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code argExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitArgExpr(DaphneDSLGrammarParser.ArgExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code colMajorFrameLiteralExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterColMajorFrameLiteralExpr(DaphneDSLGrammarParser.ColMajorFrameLiteralExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code colMajorFrameLiteralExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitColMajorFrameLiteralExpr(DaphneDSLGrammarParser.ColMajorFrameLiteralExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code callExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterCallExpr(DaphneDSLGrammarParser.CallExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code callExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitCallExpr(DaphneDSLGrammarParser.CallExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code powExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterPowExpr(DaphneDSLGrammarParser.PowExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code powExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitPowExpr(DaphneDSLGrammarParser.PowExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code identifierExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterIdentifierExpr(DaphneDSLGrammarParser.IdentifierExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code identifierExpr}
	 * labeled alternative in {@link DaphneDSLGrammarParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitIdentifierExpr(DaphneDSLGrammarParser.IdentifierExprContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#frameRow}.
	 * @param ctx the parse tree
	 */
	void enterFrameRow(DaphneDSLGrammarParser.FrameRowContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#frameRow}.
	 * @param ctx the parse tree
	 */
	void exitFrameRow(DaphneDSLGrammarParser.FrameRowContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#indexing}.
	 * @param ctx the parse tree
	 */
	void enterIndexing(DaphneDSLGrammarParser.IndexingContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#indexing}.
	 * @param ctx the parse tree
	 */
	void exitIndexing(DaphneDSLGrammarParser.IndexingContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#range}.
	 * @param ctx the parse tree
	 */
	void enterRange(DaphneDSLGrammarParser.RangeContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#range}.
	 * @param ctx the parse tree
	 */
	void exitRange(DaphneDSLGrammarParser.RangeContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#literal}.
	 * @param ctx the parse tree
	 */
	void enterLiteral(DaphneDSLGrammarParser.LiteralContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#literal}.
	 * @param ctx the parse tree
	 */
	void exitLiteral(DaphneDSLGrammarParser.LiteralContext ctx);
	/**
	 * Enter a parse tree produced by {@link DaphneDSLGrammarParser#boolLiteral}.
	 * @param ctx the parse tree
	 */
	void enterBoolLiteral(DaphneDSLGrammarParser.BoolLiteralContext ctx);
	/**
	 * Exit a parse tree produced by {@link DaphneDSLGrammarParser#boolLiteral}.
	 * @param ctx the parse tree
	 */
	void exitBoolLiteral(DaphneDSLGrammarParser.BoolLiteralContext ctx);
}