/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0

#include <libsolutil/LP.h>

#include <libsolutil/CommonData.h>
#include <libsolutil/CommonIO.h>
#include <libsolutil/StringUtils.h>
#include <libsolutil/LinearExpression.h>
#include <libsolutil/cxx20.h>

#include <liblangutil/Exceptions.h>

#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/reverse.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/tail.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/iterator/operations.hpp>

#include <boost/range/algorithm_ext/erase.hpp>

#include <optional>
#include <stack>

using namespace std;
using namespace solidity;
using namespace solidity::util;

using rational = boost::rational<bigint>;


namespace
{

/// Disjunctively combined two vectors of bools.
inline std::vector<bool>& operator|=(std::vector<bool>& _x, std::vector<bool> const& _y)
{
	solAssert(_x.size() == _y.size(), "");
	for (size_t i = 0; i < _x.size(); ++i)
		if (_y[i])
			_x[i] = true;
	return _x;
}

string toString(rational const& _x)
{
	if (_x == bigint(1) << 256)
		return "2**256";
	else if (_x == (bigint(1) << 256) - 1)
		return "2**256-1";
	else if (_x.denominator() == 1)
		return ::toString(_x.numerator());
	else
		return ::toString(_x.numerator()) + "/" + ::toString(_x.denominator());
}

string reasonToString(ReasonSet const& _reasons, size_t _minSize)
{
	auto reasonsAsStrings = _reasons | ranges::views::transform([](size_t _r) { return to_string(_r); });
	string result = "[" + joinHumanReadable(reasonsAsStrings) + "]";
	if (result.size() < _minSize)
		result.resize(_minSize, ' ');
	return result;
}


/// Removes incides set to true from a vector-like data structure.
template <class T>
void eraseIndices(T& _data, vector<bool> const& _indicesToRemove)
{
	T result;
	for (size_t i = 0; i < _data.size(); i++)
		if (!_indicesToRemove[i])
			result.push_back(move(_data[i]));
	_data = move(result);
}


void removeColumns(SolvingState& _state, vector<bool> const& _columnsToRemove)
{
	eraseIndices(_state.bounds, _columnsToRemove);
	for (Constraint& constraint: _state.constraints)
		eraseIndices(constraint.data, _columnsToRemove);
	eraseIndices(_state.variableNames, _columnsToRemove);
}

auto nonZeroEntriesInColumn(SolvingState const& _state, size_t _column)
{
	return
		_state.constraints |
		ranges::views::enumerate |
		ranges::views::filter([=](auto const& _entry) { return _entry.second.data[_column]; }) |
		ranges::views::transform([](auto const& _entry) { return _entry.first; });
}

/// @returns vectors of column- and row-indices that are connected to the given column,
/// in the sense of variables occurring in a constraint and constraints for variables.
pair<vector<bool>, vector<bool>> connectedComponent(SolvingState const& _state, size_t _column)
{
	solAssert(_state.variableNames.size() >= 2, "");

	vector<bool> includedColumns(_state.variableNames.size(), false);
	vector<bool> seenColumns(_state.variableNames.size(), false);
	vector<bool> includedRows(_state.constraints.size(), false);
	stack<size_t> columnsToProcess;
	columnsToProcess.push(_column);
	while (!columnsToProcess.empty())
	{
		size_t column = columnsToProcess.top();
		columnsToProcess.pop();
		if (includedColumns[column])
			continue;
		includedColumns[column] = true;

		for (size_t row: nonZeroEntriesInColumn(_state, column))
		{
			if (includedRows[row])
				continue;
			includedRows[row] = true;
			for (auto const& [index, entry]: _state.constraints[row].data.enumerateTail())
				if (entry && !seenColumns[index])
				{
					seenColumns[index] = true;
					columnsToProcess.push(index);
				}
		}
	}
	return make_pair(move(includedColumns), move(includedRows));
}

void normalizeRowLengths(SolvingState& _state)
{
	size_t vars = max(_state.variableNames.size(), _state.bounds.size());
	for (Constraint const& c: _state.constraints)
		vars = max(vars, c.data.size());
	_state.variableNames.resize(vars);
	_state.bounds.resize(vars);
	for (Constraint& c: _state.constraints)
		c.data.resize(vars);
}

}


bool Constraint::operator<(Constraint const& _other) const
{
	if (equality != _other.equality)
		return equality < _other.equality;

	for (size_t i = 0; i < max(data.size(), _other.data.size()); ++i)
		if (rational diff = data.get(i) - _other.data.get(i))
		{
			//cout << "Exit after " << i << endl;
			return diff < 0;
		}
	//cout << "full traversal of " << max(data.size(), _other.data.size()) << endl;

	return false;
}

bool Constraint::operator==(Constraint const& _other) const
{
	if (equality != _other.equality)
		return false;

	for (size_t i = 0; i < max(data.size(), _other.data.size()); ++i)
		if (data.get(i) != _other.data.get(i))
		{
			//cout << "Exit after " << i << endl;
			return false;
		}
	//cout << "full traversal of " << max(data.size(), _other.data.size()) << endl;

	return true;
}

bool SolvingState::Compare::operator()(SolvingState const& _a, SolvingState const& _b) const
{
	if (!considerVariableNames || _a.variableNames == _b.variableNames)
	{
		if (_a.bounds == _b.bounds)
			return _a.constraints < _b.constraints;
		else
			return _a.bounds < _b.bounds;
	}
	else
		return _a.variableNames < _b.variableNames;
}

set<size_t> SolvingState::reasons() const
{
	set<size_t> ret;
	for (Bounds const& b: bounds)
		ret += b.lowerReasons + b.upperReasons;
	return ret;
}

string SolvingState::toString() const
{
	size_t const reasonLength = 10;
	string result;
	for (Constraint const& constraint: constraints)
	{
		vector<string> line;
		for (auto&& [index, multiplier]: constraint.data.enumerate())
			if (index > 0 && multiplier != 0)
			{
				string mult =
					multiplier == -1 ?
					"-" :
					multiplier == 1 ?
					"" :
					::toString(multiplier) + " ";
				line.emplace_back(mult + variableNames.at(index));
			}
		result +=
			reasonToString(constraint.reasons, reasonLength) +
			joinHumanReadable(line, " + ") +
			(constraint.equality ? "  = " : " <= ") +
			::toString(constraint.data.front()) +
			"\n";
	}
	result += "Bounds:\n";
	for (auto&& [index, bounds]: bounds | ranges::views::enumerate)
	{
		if (!bounds.lower && !bounds.upper)
			continue;
		if (bounds.lower)
			result +=
				reasonToString(bounds.lowerReasons, reasonLength) +
				::toString(*bounds.lower) + " <= ";
		result += variableNames.at(index);
		if (bounds.upper)
			result += " <= "s + ::toString(*bounds.upper) + " " + reasonToString(bounds.upperReasons, 0);
		result += "\n";
	}
	return result;
}


void LPSolver::addConstraint(Constraint const& _constraint, optional<size_t> _reason)
{
	// TODO at this point, we could also determine if it is a fixed variable.
	// (maybe even taking the bounds on existing variables into account)
	set<size_t> touchedProblems;
	for (auto const& [index, entry]: _constraint.data.enumerateTail())
		if (entry)
			if (m_subProblemsPerVariable.count(index))
				touchedProblems.emplace(m_subProblemsPerVariable.at(index));

	if (touchedProblems.empty())
	{
		//cout << "Creating new sub problem." << endl;
		// TODO we could find an empty spot for the pointer.
		m_subProblems.emplace_back(make_shared<SubProblem>());
		solAssert(!m_subProblems.back()->sealed);
		touchedProblems.emplace(m_subProblems.size() - 1);
	}
	for (size_t problemToErase: touchedProblems | ranges::views::tail | ranges::views::reverse)
		combineSubProblems(*touchedProblems.begin(), problemToErase);
	addConstraintToSubProblem(*touchedProblems.begin(), _constraint, move(_reason));
}


pair<LPResult, variant<Model, ReasonSet>> LPSolver::check()
{
	for (auto&& [index, problem]: m_subProblems | ranges::views::enumerate)
		if (problem)
			problem->sealed = true;

	for (auto&& [index, problem]: m_subProblems | ranges::views::enumerate)
	{
		if (!problem)
			continue;
		if (!problem->result)
			problem->result = problem->check();

		if (*proble->result == LPResult::Infeasible)
			return {LPResult::Infeasible, problem->reasons};
	}
	return {LPResult::Feasible, model()};
}

LPSolver::SubProblem& LPSolver::unseal(size_t _problemIndex)
{
	solAssert(m_subProblems[_problemIndex]);
	if (m_subProblems[_problemIndex]->sealed)
		m_subProblems[_problemIndex] = make_shared<SubProblem>(*m_subProblems[_combineInto]);
	m_subProblems[_problemIndex]->sealed = false;
	m_subProblems[_problemIndex]->result = nullopt;
	return *m_subProblems[_combineInto];
}

void LPSolver::combineSubProblems(size_t _combineInto, size_t _combineFrom)
{
	SubProblem& combineInto = unseal(_combineInto);
	SubProblem const& combineFrom = *m_subProblems[_combineFrom];

	size_t varShift = combineInto.variableNames.size() - 1;
	size_t rowShift = combineInto.constraints.size();
	combineInto.constraints += combineFrom.constraints;
	combineInto.assignments += combineFrom.assignments | ranges::view::tail;
	combineInto.variableNames += combineFrom.variableNames | ranges::view::tail;
	combineInto.bounds += combineFrom.bounds | ranges::view::tail;
	for (auto&& [index, row]: combineFrom.basicVariables)
		combineInto.basicVariables.emplace(index + varShift, row + rowShift);
	for (auto&& [outerIndex, innerIndex]: combineFrom.varMapping)
		combineInto.basicVariables.emplace(outerIndex, innerIndex + varShift;
	combineInto.reasons += combineFrom.reasons;

	for (auto& item: m_subProblemsPerVariable)
		if (item.second == _combineFrom)
			item.second = _combineInto;

	m_subProblems[_combineFrom].reset();
}

// TODO move into problem and make it erturn set of vaiables added

void LPSolver::addConstraintToSubProblem(
	size_t _subProblem,
	Constraint const& _constraint,
	std::optional<size_t> _reason
)
{
	SubProblem& problem = unseal(_subProblem);
	if (_reason)
		problem.reasons.insert(*_reason);

	for (auto const& [index, entry]: _constraint.data.enumerateTail())
		if (entry && !problem.varMapping.count(index))
			addOuterVariableToSubProblem(_subProblem, index);

	size_t slackIndex = addNewVariableToSubProblem(_subProblem);
	problem.basicVariables[slackIndex] = problem.constraints.size();
	if (_constraint.equality)
		problem.variables[slackindex].bounds.lower = _constraint.data[0];
	problem.variables[slackindex].bounds.upper = _constraint.data[0];

	Constraint compressedConstraint;
	compressedConstraint.equality = true;
	compressedConstraint.data.resize(1 + problem.variables.size());
	for (auto const& [index, entry]: _constraint.data.enumerateTail())
		if (entry)
			compressedConstraint[problem.varMapping.at(index)] = entry;
	compressedConstraint[slackIndex] = -1;
	problem.constraints.emplace_back(move(compressedConstraint));
	problem.basicVariables[slackIndex] = problem.constraints.size() - 1;
}

void LPSolver::addOuterVariableToSubProblem(size_t _subProblem, size_t _outerIndex)
{
	size_t index = addNewVariableToSubProblem(_subProblem);
	problem.varMapping.emplace(_outerIndex, index);
	m_subProblemsPerVariable[_outerIndex] = _subProblem;
}

size_t LPSolver::addNewVariableToSubProblem(size_t _subProblem)
{
	SubProblem& problem = unseal(_subProblem);
	size_t index = problem.variables.size();
	for (Constraint& c: problem.constraints)
		c.data.resize(index + 1);
	problem.variables.emplace_back();
	return index;
}

map<string, rational> LPSolver::model() const
{
	map<string, rational> result;
	for (auto const& problem: m_subProblems)
		if (problem)
			for (auto&& [outerIndex, innerIndex]: problem->varMapping)
				result[problem->variables[innerIndex].name] = problem->variables[innerIndex].value;
	return result;
}

LPResult LPSolver::SubProblem::check()
{

	// Adjust the assignments so we satisfy the bounds of the non-basic variables.
	for (auto const& [i, var]: variables | ranges::views::enumerate | ranges::views::tail)
	{
		if (basicVariables.count(i) || (!var.bounds.lower && !var.bounds.upper))
			continue;
		if (var.bounds.lower && var.bounds.upper)
			solAssert(*var.bounds.lower <= *var.bounds.upper);
		if (var.bounds.lower && var.value < *bounds.lower)
			update(i, *bounds.lower);
		else if (bounds.upper && var.value > *bounds.upper)
			update(i, *bounds.upper);
	}

	// Now try to make the basic variables happy, pivoting if necessary.

	// TODO bound number of iterations
	while (auto bvi = firstConflictingBasicVariable())
	{
		if (variables[*bvi].bounds.lower && variables[*bvi].value < *variables[*bvi].bounds.lower)
		{
			if (auto replacementVar = firstReplacementVar(*bvi, true))
				pivotAndUpdate(*bvi, *variables[*bvi].bounds.lower, *replacementVar);
			else
				return LPResult::Infeasible;
		}
		else if (variables[*bvi].bounds.upper && variables[*bvi].value > *variables[*bvi].bounds.upper)
		{
			if (auto replacementVar = firstReplacementVar(*bvi, false))
				pivotAndUpdate(*bvi, *variables[*bvi].bounds.upper, *replacementVar);
			else
				return LPResult::Infeasible;
		}
	}

	return LPResult::Feasible;
}

void LPSolver::SubProblem::update(size_t _varIndex, rational const& _value)
{
	rational delta = _value - variables[_varIndex].value;
	variables[_varIndex].value = _value;
	for (size_t j = 1; j < variables.size(); j++)
		if (basicVariables.count(j) && constraints[basicVariables.at(j)].data[_varIndex])
			variables[j].value += delta * constraints[basicVariables.at(j)].data[_varIndex];

}

optional<size_t> LPSolver::SubProblem::firstConflictingBasicVariable() const
{
	for (auto const& varItem: basicVariables)
	{
		Variable const& variable = variables[varItem.first];
		if (
			(variable.bounds.lower && variable.value < *variable.bounds.lower) ||
			(variable.bounds.upper && variable.value > *variable.bounds.upper)
		)
			return i;
	}
	return nullopt;
}

optional<size_t> LPSolver::SubProblem::firstReplacementVar(
	size_t _basicVarToReplace,
	bool _increasing
) const
{
	LinearExpression const& basicVarEquation = constraints[basicVariables.at(_basicVarToReplace)].data;
	for (auto const& [i, var]: variables | ranges::views::enumerate | ranges::views::tail)
	{
		if (basicVariables.count(i) || !basicVarEquation[i])
			continue;
		bool positive = basicVarEquation[i] > 0;
		if (!_increasing)
			positive = !positive;
		Variable const& candidate = variables[i];
		if (positive && (!candidate.bounds.upper || candidate.value < candidate.bounds.upper))
			return i;
		if (!positive && (!candidate.bounds.lower || candidate.value > candidate.bounds.lower)
			return i;
	}
	return nullopt;
}

void LPSolver::SubProblem::pivot(size_t _old, size_t _new)
{
	// Transform pivotRow such that the coefficient for _new is -1
	// Then use that to set all other coefficients for _new to zero.
	size_t pivotRow = basicVariables[_old];
	LinearExpression& pivotRowData = constraints[pivotRow].data;

	rational pivot = pivotRowData[_new];
	solAssert(pivot != 0, "");
	if (pivot != -1)
		pivotRowData /= -pivot;
	solAssert(pivotRowData[_new] == rational(-1), "");

	auto subtractMultipleOfPivotRow = [&](LinearExpression& _row) {
		if (_row[_new] == 0)
			return;
		else if (_row[_new] == rational{1})
			_row += pivotRowData;
		else if (_row[_new] == rational{-1})
			_row -= pivotRowData;
		else
			_row += _row[_new] * pivotRowData;
	};

	for (size_t i = 0; i < constraints.size(); ++i)
		if (i != pivotRow)
			subtractMultipleOfPivotRow(constraints[i].data);

	basicVariables.erase(_old);
	basicVariables[_new] = pivotRow;
}

void LPSolver::SubProblem::pivotAndUpdate(
	size_t _oldBasicVar,
	rational const& _newValue,
	size_t _newBasicVar
)
{
	rational theta = (_newValue - variables[_oldBasicVar].value) / constraints[basicVariables[_oldBasicVar]].data[_newBasicVar];

	variables[_oldBasicVar].value = _newValue;
	variables[_newBasicVar].value += theta;

	for (auto const& [i, row]: basicVariables)
		if (i != _oldBasicVar && constraints[row].data[_newBasicVar])
			variables[i].value += constraints[row].data[_newBasicVar] * theta;

	pivot(_oldBasicVar, _newBasicVar);
	//cout << "After pivot and update: " << endl << toString() << endl;
}
